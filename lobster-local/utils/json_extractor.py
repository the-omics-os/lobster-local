"""
JSON extraction utilities for handling LLM responses that may contain extra text.

This module provides robust JSON parsing that can extract valid JSON from mixed content,
which is common when LLMs add explanatory text before or after the requested JSON.
"""

import json
import re
from typing import Dict, Any, Optional
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def extract_json_from_text(text: str, fallback_response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract valid JSON from text that may contain additional content.
    
    This function tries multiple strategies to find and parse JSON:
    1. Direct JSON parsing (if the text is pure JSON)
    2. Find JSON objects within the text using regex
    3. Find JSON objects within code blocks
    4. Return fallback response if all parsing fails
    
    Args:
        text: The text content that should contain JSON
        fallback_response: Default response to return if JSON extraction fails
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        ValueError: If no valid JSON found and no fallback provided
    """
    if not text or not text.strip():
        if fallback_response is not None:
            return fallback_response
        raise ValueError("Empty text provided and no fallback response")
    
    text = text.strip()
    
    # Strategy 1: Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parsing failed: {e}")
    
    # Strategy 2: Find JSON objects using regex patterns
    json_patterns = [
        # Match JSON objects (including nested)
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        # Match JSON with more complex nesting
        r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}',
        # Match JSON arrays
        r'\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]'
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                logger.debug(f"Successfully extracted JSON using pattern: {pattern[:20]}...")
                return parsed
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Look for JSON within code blocks or specific markers
    code_block_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'`(\{.*?\})`',
        r'<json>\s*(\{.*?\})\s*</json>',
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                logger.debug("Successfully extracted JSON from code block")
                return parsed
            except json.JSONDecodeError:
                continue
    
    # Strategy 4: Try to find lines that look like JSON
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                parsed = json.loads(line)
                logger.debug("Successfully extracted JSON from line")
                return parsed
            except json.JSONDecodeError:
                continue
    
    # Strategy 5: Try to extract JSON from the beginning or end of text
    for start_idx in [0, text.find('{')]:
        if start_idx >= 0:
            for end_idx in [len(text), text.rfind('}') + 1]:
                if end_idx > start_idx:
                    try:
                        candidate = text[start_idx:end_idx].strip()
                        if candidate.startswith('{') and candidate.endswith('}'):
                            parsed = json.loads(candidate)
                            logger.debug("Successfully extracted JSON from substring")
                            return parsed
                    except json.JSONDecodeError:
                        continue
    
    # All strategies failed
    logger.warning(f"Failed to extract JSON from text: {text[:200]}...")
    
    if fallback_response is not None:
        logger.info("Using fallback response for JSON extraction failure")
        return fallback_response
    
    raise ValueError(f"Could not extract valid JSON from text: {text[:100]}...")


def safe_json_parse(text: str, expected_keys: Optional[list] = None) -> Dict[str, Any]:
    """
    Safely parse JSON with validation of expected keys.
    
    Args:
        text: Text to parse as JSON
        expected_keys: List of keys that should be present in the parsed JSON
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If parsing fails or expected keys are missing
    """
    try:
        result = extract_json_from_text(text)
        
        # Validate expected keys if provided
        if expected_keys:
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                raise ValueError(f"Missing expected keys in JSON: {missing_keys}")
        
        return result
        
    except Exception as e:
        logger.error(f"Safe JSON parsing failed: {e}")
        raise


def create_fallback_clarification_response(question: str = None) -> Dict[str, Any]:
    """
    Create a fallback response for clarification requests.
    
    Args:
        question: Optional custom question to ask
        
    Returns:
        Standard clarification response format
    """
    default_question = (
        "I need more information to help with your bioinformatics analysis. "
        "Could you please provide more details about what you'd like to analyze?"
    )
    
    return {
        "need_clarification": True,
        "question": question or default_question,
        "verification": ""
    }


def create_fallback_no_clarification_response() -> Dict[str, Any]:
    """
    Create a fallback response when no clarification is needed.
    
    Returns:
        Standard no-clarification response format
    """
    return {
        "need_clarification": False,
        "question": "",
        "verification": "I'll proceed with the bioinformatics analysis based on the information provided."
    }
