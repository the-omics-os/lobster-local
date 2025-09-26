"""
Research Agent Assistant for metadata validation and LLM-based operations.

This module handles metadata validation for datasets to help the Research Agent
quickly identify datasets with required fields before downloading, saving time
and computational resources.
"""

import json
import re
from typing import Dict, Any, Optional, List
from langchain_aws import ChatBedrockConverse
from pydantic import Field, BaseModel, validator

from lobster.config.settings import get_settings
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataValidationConfig(BaseModel):
    """Configuration for dataset metadata validation results."""
    has_required_fields: bool = Field(
        description="Whether all required fields are present in the dataset"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of required fields that are missing"
    )
    available_fields: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Available fields with sample values (field_name -> sample values)"
    )
    sample_count_by_field: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of samples that have each field (field_name -> count)"
    )
    total_samples: int = Field(
        default=0,
        description="Total number of samples in the dataset"
    )
    field_coverage: Dict[str, float] = Field(
        default_factory=dict,
        description="Percentage of samples with each field (field_name -> percentage)"
    )
    recommendation: str = Field(
        default="manual_check",
        description="Recommendation: 'proceed' | 'skip' | 'manual_check'"
    )
    confidence_score: float = Field(
        default=0.0,
        description="Confidence score for metadata quality (0-1)"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings or issues found during validation"
    )
    
    @validator('recommendation')
    def validate_recommendation(cls, v):
        allowed = {'proceed', 'skip', 'manual_check'}
        if v not in allowed:
            raise ValueError(f"recommendation must be one of {allowed}")
        return v
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("confidence_score must be between 0 and 1")
        return v


class FieldMapping(BaseModel):
    """Mapping configuration for field name variations."""
    canonical_name: str = Field(description="The standardized field name")
    variations: List[str] = Field(
        default_factory=list,
        description="Common variations of the field name"
    )
    value_mappings: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping of canonical values to their variations"
    )


class ResearchAgentAssistant:
    """Assistant class for handling metadata validation for Research Agent."""
    
    # Common field name variations for normalization
    FIELD_MAPPINGS = {
        "smoking_status": FieldMapping(
            canonical_name="smoking_status",
            variations=[
                "smoking status", "smoker", "smoking history", 
                "tobacco use", "cigarette smoking", "smoking"
            ],
            value_mappings={
                "current": ["current smoker", "smoker", "yes", "active"],
                "former": ["former smoker", "ex-smoker", "past", "quit"],
                "never": ["never smoker", "non-smoker", "no", "never"]
            }
        ),
        "treatment_response": FieldMapping(
            canonical_name="treatment_response",
            variations=[
                "response", "treatment outcome", "clinical response",
                "therapeutic response", "drug response", "response to treatment"
            ],
            value_mappings={
                "responder": ["R", "response", "sensitive", "good response"],
                "non_responder": ["NR", "no response", "resistant", "poor response"],
                "partial": ["PR", "partial response", "intermediate"]
            }
        ),
        "cancer_stage": FieldMapping(
            canonical_name="cancer_stage",
            variations=[
                "stage", "tumor stage", "clinical stage", 
                "pathological stage", "tnm stage", "disease stage"
            ],
            value_mappings={
                "I": ["1", "stage 1", "stage i", "early"],
                "II": ["2", "stage 2", "stage ii"],
                "III": ["3", "stage 3", "stage iii", "advanced"],
                "IV": ["4", "stage 4", "stage iv", "metastatic"]
            }
        ),
        "timepoint": FieldMapping(
            canonical_name="timepoint",
            variations=[
                "time point", "time", "collection time", 
                "sample time", "visit", "day"
            ]
        ),
        "mutation_status": FieldMapping(
            canonical_name="mutation_status",
            variations=[
                "mutation", "genotype", "variant", 
                "genetic status", "mutational status"
            ]
        )
    }
    
    def __init__(self):
        """Initialize the Research Agent Assistant."""
        self.settings = get_settings()
        self._llm = None
    
    @property
    def llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            llm_params = self.settings.get_agent_llm_params('assistant')
            self._llm = ChatBedrockConverse(**llm_params)
        return self._llm
    
    def normalize_field_name(self, field: str) -> str:
        """Normalize field name to canonical form."""
        field_lower = field.lower().strip()
        
        for canonical, mapping in self.FIELD_MAPPINGS.items():
            if field_lower == canonical or field_lower in mapping.variations:
                return canonical
        
        return field_lower
    
    def normalize_field_value(self, field: str, value: str) -> str:
        """Normalize field value to canonical form."""
        field_norm = self.normalize_field_name(field)
        value_lower = value.lower().strip()
        
        if field_norm in self.FIELD_MAPPINGS:
            mapping = self.FIELD_MAPPINGS[field_norm]
            for canonical_value, variations in mapping.value_mappings.items():
                if value_lower == canonical_value or value_lower in variations:
                    return canonical_value
        
        return value_lower
    
    def validate_dataset_metadata(
        self, 
        metadata: Dict[str, Any],
        geo_id: str,
        required_fields: List[str],
        required_values: Optional[Dict[str, List[str]]] = None,
        threshold: float = 0.8
    ) -> Optional[MetadataValidationConfig]:
        """
        Validate dataset metadata using LLM to check for required fields.
        
        Args:
            metadata: GEO dataset metadata dictionary
            geo_id: GEO accession number
            required_fields: List of required field names
            required_values: Optional dict of field -> required values
            threshold: Minimum fraction of samples that must have the field
            
        Returns:
            MetadataValidationConfig object if successful, None otherwise
        """
        try:
            # Normalize required fields
            normalized_required = [self.normalize_field_name(f) for f in required_fields]
            
            # Extract key metadata sections
            title = metadata.get('title', 'N/A')
            summary = metadata.get('summary', 'N/A')
            overall_design = metadata.get('overall_design', 'N/A')
            characteristics = metadata.get('characteristics_ch1', [])
            sample_count = metadata.get('n_samples', 0)
            
            # Get the schema from MetadataValidationConfig
            validation_schema = MetadataValidationConfig.model_json_schema()
            
            # Create a focused context for the LLM
            metadata_context = f"""
Dataset: {geo_id}
Title: {title}
Sample Count: {sample_count}

Summary: {summary}

Overall Design: {overall_design}

Sample Characteristics (first 5 samples):
{json.dumps(characteristics[:5], indent=2) if characteristics else 'No characteristics data available'}
"""

            # Create highly structured system prompt with explicit instructions
            system_prompt = f"""You are a bioinformatics metadata validation expert. Your task is to analyze GEO dataset metadata and validate the presence of required fields.

CRITICAL INSTRUCTIONS FOR ACCURATE ANALYSIS:
1. You MUST return ONLY a valid JSON object - no markdown, no explanations, no additional text
2. Carefully examine the metadata for field names and their variations
3. Count ACTUAL samples that have each field, not just whether the field exists
4. Be conservative in your assessment - only mark fields as present if they clearly exist
5. If uncertain about a field, mark it as missing and add a warning

Field Name Normalization Rules:
- "smoking status" = "smoking_status" = "smoker" = "tobacco use"
- "treatment response" = "response" = "clinical response" = "therapeutic response"
- "stage" = "cancer_stage" = "tumor stage" = "clinical stage"
- "time" = "timepoint" = "time point" = "collection time"
- "mutation" = "mutation_status" = "genotype" = "variant"

The output schema MUST conform exactly to:
{json.dumps(validation_schema, indent=2)}

Recommendation Logic:
- "proceed": All required fields present with >= {threshold*100}% coverage
- "skip": Missing critical fields OR any field has < 50% coverage
- "manual_check": Some fields present but coverage between 50-{threshold*100}%

Remember: Return ONLY the JSON object, nothing else."""

            # Create the validation prompt
            prompt = f"""Analyze this GEO dataset metadata to validate these required fields:
{json.dumps(normalized_required)}

{f"Required values for fields: {json.dumps(required_values)}" if required_values else ""}

Metadata to analyze:
{metadata_context}

IMPORTANT: 
1. Check each required field for actual presence in sample characteristics
2. Count how many samples have each field (not just if it exists somewhere)
3. Calculate coverage percentage for each field
4. Return ONLY a valid JSON object matching the schema
"""

            # Invoke the LLM with structured output expectation
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])
            
            # Extract the JSON from the response
            response_text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content)
            
            # Clean the response text to extract JSON
            # Remove any markdown code blocks or extra text
            response_text = response_text.strip()
            if response_text.startswith('```'):
                # Extract content between ```json and ```
                json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
            
            # Try to find JSON object in the text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            # Parse the JSON response
            try:
                validation_dict = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response: {e}")
                logger.debug(f"Response text: {response_text[:500]}...")
                
                # Create a fallback validation with warnings
                validation_dict = {
                    "has_required_fields": False,
                    "missing_fields": required_fields,
                    "available_fields": {},
                    "sample_count_by_field": {},
                    "total_samples": sample_count,
                    "field_coverage": {},
                    "recommendation": "manual_check",
                    "confidence_score": 0.0,
                    "warnings": ["LLM response parsing failed - manual validation recommended"]
                }
            
            # Ensure all required schema fields are present
            validation_dict.setdefault("has_required_fields", False)
            validation_dict.setdefault("missing_fields", [])
            validation_dict.setdefault("available_fields", {})
            validation_dict.setdefault("sample_count_by_field", {})
            validation_dict.setdefault("total_samples", sample_count)
            validation_dict.setdefault("field_coverage", {})
            validation_dict.setdefault("recommendation", "manual_check")
            validation_dict.setdefault("confidence_score", 0.0)
            validation_dict.setdefault("warnings", [])
            
            # Create MetadataValidationConfig object
            validation_config = MetadataValidationConfig(**validation_dict)
            
            # Post-process to ensure required values are checked if specified
            if required_values and validation_config.has_required_fields:
                for field, required_vals in required_values.items():
                    field_norm = self.normalize_field_name(field)
                    if field_norm in validation_config.available_fields:
                        available_vals = validation_config.available_fields[field_norm]
                        # Normalize available values
                        normalized_available = [
                            self.normalize_field_value(field_norm, v) 
                            for v in available_vals
                        ]
                        # Check if any required values are present
                        required_vals_norm = [
                            self.normalize_field_value(field_norm, v) 
                            for v in required_vals
                        ]
                        if not any(v in normalized_available for v in required_vals_norm):
                            validation_config.warnings.append(
                                f"Field '{field}' exists but doesn't contain required values: {required_vals}"
                            )
                            if validation_config.recommendation == "proceed":
                                validation_config.recommendation = "manual_check"
                                validation_config.confidence_score *= 0.8
            
            logger.info(f"Successfully validated metadata for {geo_id}")
            logger.debug(f"Validation result: {validation_config.recommendation}")
            return validation_config
                            
        except Exception as e:
            logger.warning(f"Failed to validate metadata for {geo_id}: {e}")
            
            # Return a conservative validation result on error
            return MetadataValidationConfig(
                has_required_fields=False,
                missing_fields=required_fields,
                available_fields={},
                sample_count_by_field={},
                total_samples=metadata.get('n_samples', 0),
                field_coverage={},
                recommendation="manual_check",
                confidence_score=0.0,
                warnings=[f"Validation error: {str(e)}"]
            )
    
    def format_validation_report(self, validation_config: MetadataValidationConfig, geo_id: str) -> str:
        """
        Format the validation configuration into a readable report.
        
        Args:
            validation_config: MetadataValidationConfig object
            geo_id: GEO accession number
            
        Returns:
            Formatted string report
        """
        config_dict = validation_config.model_dump()
        
        # Determine status emoji
        status_emoji = {
            "proceed": "✅",
            "skip": "❌",
            "manual_check": "⚠️"
        }.get(config_dict.get('recommendation', 'manual_check'), "❓")
        
        report = f"""
## Metadata Validation Report for {geo_id}

**Recommendation:** {status_emoji} **{config_dict.get('recommendation', 'Unknown').upper()}**
**Confidence Score:** {config_dict.get('confidence_score', 0):.2f}/1.00
**Total Samples:** {config_dict.get('total_samples', 0)}

### Field Analysis:
"""
        
        # Add field coverage details
        field_coverage = config_dict.get('field_coverage', {})
        for field, coverage in field_coverage.items():
            status = "✅" if coverage >= 80 else "⚠️" if coverage >= 50 else "❌"
            report += f"- **{field}**: {status} {coverage:.1f}% coverage "
            
            # Add sample values if available
            if field in config_dict.get('available_fields', {}):
                values = config_dict.get('available_fields', {}).get(field, [])[:3]
                if values:
                    report += f"(values: {', '.join(repr(v) for v in values)})"
            report += "\n"
        
        # Add missing fields
        missing = config_dict.get('missing_fields', [])
        if missing:
            report += "\n### Missing Required Fields:\n"
            for field in missing:
                report += f"- ❌ {field}\n"
        
        # Add warnings
        warnings = config_dict.get('warnings', [])
        if warnings:
            report += "\n### ⚠️ Warnings:\n"
            for warning in warnings:
                report += f"- {warning}\n"
        
        # Add recommendation explanation
        report += "\n### 💡 Recommendation Rationale:\n"
        if config_dict.get('recommendation') == 'proceed':
            report += "All required fields are present with sufficient coverage. Dataset is suitable for analysis.\n"
        elif config_dict.get('recommendation') == 'skip':
            report += "Critical fields are missing or have insufficient coverage. Consider finding alternative datasets.\n"
        else:
            report += "Some required fields are present but coverage is limited. Manual inspection recommended to assess usability.\n"
        
        return report
