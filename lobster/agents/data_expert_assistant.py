"""
Data Expert Assistant for LLM-based operations.

This module handles all LLM-based strategy extraction and decision making
for the Data Expert Agent, keeping the main agent file clean and focused
on data operations.
"""

import json
import re
from typing import Dict, Any, Optional, List
from langchain_aws import ChatBedrockConverse
from pydantic import Field, BaseModel

from lobster.config.settings import get_settings
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class StrategyConfig(BaseModel):
    """Configuration for GEO dataset download strategy."""
    summary_file_name: str = Field(
        default="", 
        description="name of the summary or overview file (has 'GSE' in the name). Example: 'GSE131907_Lung_Cancer_Feature_Summary'"
    )
    summary_file_type: str = Field(
        default="", 
        description="Filetype of the Summary File. Example: 'xlsx'"
    )
    processed_matrix_name: str = Field(
        default="", 
        description="name of the TARGET processed (log2, tpm, normalized, etc) matrix file (preference is non R objects). Example: 'GSE131907_Lung_Cancer_normalized_log2TPM_matrix'"
    )
    processed_matrix_filetype: str = Field(
        default="", 
        description="filetype of the TARGET processed file (preference is non R objects). Example: 'txt'"
    )
    raw_UMI_like_matrix_name: str = Field(
        default="", 
        description="name of the raw TARGET UMI-like (UMI, raw, tar) matrix file (preference is non R objects). Example: 'GSE131907_Lung_Cancer_raw_UMI_matrix'"
    )
    raw_UMI_like_matrix_filetype: str = Field(
        default="", 
        description="filetype of the TARGET raw UMI-like file (preference is non R objects). Example: 'txt'"
    )
    cell_annotation_name: str = Field(
        default="", 
        description="Filename of the file which likely is linked to the cell annotation. Example:'GSE131907_Lung_Cancer_cell_annotation'"
    )
    cell_annotation_filetype: str = Field(
        default="", 
        description="Filetype of cell annotation file. Example: 'txt'"
    )
    raw_data_available: bool = Field(
        default=False, 
        description="If based on the metadata raw data is available for the samples. Sometimes it stated in the study description"
    )


class DataExpertAssistant:
    """Assistant class for handling LLM-based operations for Data Expert."""
    
    def __init__(self):
        """Initialize the Data Expert Assistant."""
        self.settings = get_settings()
        self._llm = None
    
    def _sanitize_null_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize various null value representations to ensure consistency.
        
        Args:
            data: Dictionary that may contain various null representations
            
        Returns:
            Dictionary with null values properly sanitized
        """
        sanitized = {}
        
        for key, value in data.items():
            # Handle various null representations
            if value is None:
                # Actual Python None
                if key == 'raw_data_available':
                    sanitized[key] = False
                else:
                    sanitized[key] = "NA"
            elif isinstance(value, str):
                # Handle string representations of null
                null_strings = ['null', 'None', 'NULL', 'none', 'nil', 'NIL', 'n/a', 'N/A', 'na', 'NA']
                if value.strip() in null_strings or value.strip() == "":
                    if key == 'raw_data_available':
                        sanitized[key] = False
                    else:
                        sanitized[key] = "NA"
                else:
                    sanitized[key] = value.strip()
            elif isinstance(value, bool):
                # Boolean values are fine as-is
                sanitized[key] = value
            else:
                # Convert any other type to string, but check for null-like values
                str_value = str(value).strip()
                null_strings = ['null', 'None', 'NULL', 'none', 'nil', 'NIL', 'n/a', 'N/A', 'na', 'NA']
                if str_value in null_strings or str_value == "":
                    if key == 'raw_data_available':
                        sanitized[key] = False
                    else:
                        sanitized[key] = "NA"
                else:
                    sanitized[key] = str_value
        
        return sanitized
    
    @property
    def llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            llm_params = self.settings.get_agent_llm_params('assistant')
            self._llm = ChatBedrockConverse(**llm_params)
        return self._llm
    
    def extract_strategy_config(
        self, 
        metadata: Dict[str, Any],
        geo_id: str
    ) -> Optional[StrategyConfig]:
        """
        Extract strategy configuration from GEO metadata using LLM.
        
        Args:
            metadata: GEO dataset metadata dictionary
            geo_id: GEO accession number
            
        Returns:
            StrategyConfig object if successful, None otherwise
        """
        try:
            # Extract key metadata fields
            title = metadata.get('title', 'N/A')
            summary = metadata.get('summary', 'N/A')
            overall_design = metadata.get('overall_design', 'N/A')
            supplementary_files = metadata.get('supplementary_file', [])
            
            # Prepare the context for LLM
            metadata_context = f"""
Title: {title}

Summary: {summary}

Overall Design: {overall_design}

Supplementary Files:
{supplementary_files if supplementary_files else 'No supplementary files listed'}
"""
            
            # Get the schema from StrategyConfig class
            strategy_schema = StrategyConfig.model_json_schema()
            
            # Create system prompt using the schema
            system_prompt = f"""You are a bioinformatics expert analyzing GEO dataset metadata. Your task is to extract file information and populate a StrategyConfig object.

The StrategyConfig schema is:
{json.dumps(strategy_schema, indent=2)}

Important notes:
1. For each file type, extract the filename WITHOUT the extension (e.g., "GSE131907_Lung_Cancer_Feature_Summary" not "GSE131907_Lung_Cancer_Feature_Summary.xlsx")
2. Extract the file extension separately (e.g., "xlsx", "txt", "csv")
3. Prefer non-R objects (.txt, .csv) over .rds files when multiple options exist
4. Check the summary and overall design text for mentions of raw data availability
5. If a field cannot be determined from the metadata, use an empty string "" for string fields and false for boolean fields
6. DO NOT use null, None, or any null-like values - use empty strings for unknown string fields

Return only a valid JSON object that matches the StrategyConfig schema."""

            # Create the prompt
            prompt = f"""Given this GEO dataset metadata, extract the file information into a StrategyConfig:

{metadata_context}

Return only a valid JSON object that conforms to the StrategyConfig schema provided in the system prompt.
"""

            # Invoke the LLM
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])
            
            # Extract the JSON from the response
            response_text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content)
            
            # Parse the JSON response
            # Try to extract JSON from the response in case it's wrapped in markdown or other text
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                strategy_dict = json.loads(json_match.group())
            else:
                strategy_dict = json.loads(response_text)
            
            # Sanitize null values before creating StrategyConfig
            sanitized_dict = self._sanitize_null_values(strategy_dict)
            
            # Create StrategyConfig object
            strategy_config = StrategyConfig(**sanitized_dict)
            
            logger.info(f"Successfully extracted strategy config for {geo_id}")
            return strategy_config
                            
        except Exception as e:
            logger.warning(f"Failed to extract strategy config using LLM for {geo_id}: {e}")
            return None
    
    def _format_file_display(self, filename: str, filetype: str) -> str:
        """
        Format filename and filetype for display, handling NA values.
        
        Args:
            filename: The filename (may be "NA")
            filetype: The filetype (may be "NA")
            
        Returns:
            Formatted string for display
        """
        # Handle various null-like values
        if not filename or filename in ["NA", "Not found", ""]:
            return "Not found"
        
        if not filetype or filetype in ["NA", "Not found", ""]:
            return filename
        
        return f"{filename}.{filetype}"
    
    def format_strategy_section(self, strategy_config: StrategyConfig) -> str:
        """
        Format the strategy configuration into a readable section.
        
        Args:
            strategy_config: StrategyConfig object
            
        Returns:
            Formatted string section
        """
        config_dict = strategy_config.model_dump()
        
        # Format each file type with proper null handling
        summary_file = self._format_file_display(
            config_dict.get('summary_file_name', ''),
            config_dict.get('summary_file_type', '')
        )
        
        processed_matrix = self._format_file_display(
            config_dict.get('processed_matrix_name', ''),
            config_dict.get('processed_matrix_filetype', '')
        )
        
        raw_matrix = self._format_file_display(
            config_dict.get('raw_UMI_like_matrix_name', ''),
            config_dict.get('raw_UMI_like_matrix_filetype', '')
        )
        
        cell_annotations = self._format_file_display(
            config_dict.get('cell_annotation_name', ''),
            config_dict.get('cell_annotation_filetype', '')
        )
        
        strategy_section = f"""

📁 **Extracted File Strategy Configuration:**
- **Summary File:** {summary_file}
- **Processed Matrix:** {processed_matrix}
- **Raw/UMI Matrix:** {raw_matrix}
- **Cell Annotations:** {cell_annotations}
- **Raw Data Available:** {'Yes' if config_dict.get('raw_data_available', False) else 'No (see study description for details)'}
"""
        
        return strategy_section
    
    def _has_valid_file(self, filename: str) -> bool:
        """
        Check if a filename represents a valid file (not null/NA/empty).
        
        Args:
            filename: The filename to check
            
        Returns:
            True if filename is valid, False otherwise
        """
        if not filename:
            return False
        
        null_values = ["NA", "N/A", "na", "n/a", "null", "None", "NULL", "none", "", " "]
        return filename.strip() not in null_values
    
    def analyze_download_strategy(
        self, 
        strategy_config: StrategyConfig,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the strategy configuration to provide download recommendations.
        
        Args:
            strategy_config: Extracted strategy configuration
            metadata: Original metadata
            
        Returns:
            Dictionary with analysis results and recommendations
        """
        analysis = {
            "has_processed_matrix": self._has_valid_file(strategy_config.processed_matrix_name),
            "has_raw_matrix": self._has_valid_file(strategy_config.raw_UMI_like_matrix_name),
            "has_cell_annotations": self._has_valid_file(strategy_config.cell_annotation_name),
            "has_summary": self._has_valid_file(strategy_config.summary_file_name),
            "raw_data_available": strategy_config.raw_data_available,
            "recommendations": []
        }
        
        # Generate recommendations based on available files
        if analysis["has_processed_matrix"]:
            analysis["recommendations"].append(
                "MATRIX_FIRST: Use processed matrix for quick analysis (may include preprocessing decisions)"
            )
        
        if analysis["has_raw_matrix"] or analysis["raw_data_available"]:
            analysis["recommendations"].append(
                "SAMPLES_FIRST: Start from raw data for full control over preprocessing"
            )
        
        # Check for H5AD files in supplementary files
        supp_files = metadata.get('supplementary_file', [])
        has_h5ad = any('.h5ad' in str(f).lower() for f in supp_files)
        
        if has_h5ad:
            analysis["has_h5ad"] = True
            analysis["recommendations"].insert(
                0, "H5_FIRST: Use H5AD file for pre-processed single-cell data with metadata"
            )
        else:
            analysis["has_h5ad"] = False
        
        # Default recommendation if no clear strategy
        if not analysis["recommendations"]:
            analysis["recommendations"].append(
                "AUTO: Let the system determine the best approach based on available files"
            )
        
        return analysis
