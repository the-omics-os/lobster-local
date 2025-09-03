"""
Lobster Cloud Lambda Function - AWS Backend for Lobster AI
Minimal production-ready implementation for business validation
"""

import json
import os
import sys
import traceback
from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Hardcoded API keys for testing (in production, use AWS Secrets Manager)
VALID_API_KEYS = {
    "test-enterprise-001": {
        "name": "Test Enterprise User 1",
        "tier": "enterprise",
        "max_queries_per_hour": 100
    },
    "test-enterprise-002": {
        "name": "Test Enterprise User 2", 
        "tier": "enterprise",
        "max_queries_per_hour": 100
    },
    "demo-user-001": {
        "name": "Demo User",
        "tier": "demo",
        "max_queries_per_hour": 10
    }
}

def validate_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Validate the provided API key
    
    Args:
        api_key: The API key to validate
        
    Returns:
        User info if valid, None if invalid
    """
    return VALID_API_KEYS.get(api_key)

def get_lobster_client():
    """
    Initialize the local Lobster client
    Import here to handle cold starts better
    """
    try:
        # Import lobster-local components
        from lobster_local.core.client import LobsterClient
        from lobster_local.core.data_manager_v2 import DataManagerV2
        
        # Initialize with minimal configuration
        data_manager = DataManagerV2(
            cache_dir="/tmp/lobster_cache"  # Use Lambda tmp directory
        )
        
        client = LobsterClient(data_manager=data_manager)
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize Lobster client: {e}")
        logger.error(traceback.format_exc())
        raise

def lambda_handler(event, context):
    """
    AWS Lambda handler function
    
    Args:
        event: AWS Lambda event (API Gateway request)
        context: AWS Lambda context
        
    Returns:
        AWS API Gateway response
    """
    
    # CORS headers for browser compatibility
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }
    
    try:
        # Handle preflight OPTIONS request
        if event.get("httpMethod") == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({"message": "CORS preflight"})
            }
        
        # Log the incoming request (without sensitive data)
        logger.info(f"Received request: {event.get('httpMethod')} {event.get('path')}")
        
        # Parse the request
        if event.get("httpMethod") != "POST":
            return {
                "statusCode": 405,
                "headers": headers,
                "body": json.dumps({
                    "error": "Method not allowed. Use POST.",
                    "success": False
                })
            }
        
        # Get API key from Authorization header
        auth_header = event.get("headers", {}).get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return {
                "statusCode": 401,
                "headers": headers,
                "body": json.dumps({
                    "error": "Missing or invalid Authorization header. Use 'Bearer <api_key>'",
                    "success": False
                })
            }
        
        api_key = auth_header[7:]  # Remove "Bearer " prefix
        
        # Validate API key
        user_info = validate_api_key(api_key)
        if not user_info:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            return {
                "statusCode": 401,
                "headers": headers,
                "body": json.dumps({
                    "error": "Invalid API key",
                    "success": False
                })
            }
        
        # Parse request body
        try:
            if isinstance(event.get("body"), str):
                body = json.loads(event["body"])
            else:
                body = event.get("body", {})
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps({
                    "error": "Invalid JSON in request body",
                    "success": False
                })
            }
        
        # Route the request based on path
        path = event.get("path", "").rstrip("/")
        
        if path == "/query" or path == "":
            return handle_query(body, user_info, headers)
        elif path == "/status":
            return handle_status(headers)
        elif path == "/usage":
            return handle_usage(api_key, user_info, headers)
        elif path == "/models":
            return handle_models(headers)
        else:
            return {
                "statusCode": 404,
                "headers": headers,
                "body": json.dumps({
                    "error": f"Endpoint not found: {path}",
                    "success": False
                })
            }
            
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({
                "error": "Internal server error",
                "success": False,
                "request_id": context.aws_request_id if context else "unknown"
            })
        }

def handle_query(body: Dict[str, Any], user_info: Dict[str, Any], headers: Dict[str, str]):
    """
    Handle query requests
    """
    try:
        # Validate query input
        query = body.get("query", "").strip()
        if not query:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps({
                    "error": "Missing or empty 'query' field",
                    "success": False
                })
            }
        
        # Log the query (for usage analytics)
        logger.info(f"Processing query for user {user_info['name']}: {query[:100]}...")
        
        # Initialize Lobster client
        client = get_lobster_client()
        
        # Process the query using local Lobster logic
        options = body.get("options", {})
        result = client.query(query, **options)
        
        # Ensure result is JSON serializable
        if hasattr(result, 'to_dict'):
            result = result.to_dict()
        elif not isinstance(result, dict):
            result = {"response": str(result), "success": True}
        
        # Add cloud metadata
        result["cloud_processed"] = True
        result["user_tier"] = user_info["tier"]
        result["success"] = True
        
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({
                "error": f"Query processing failed: {str(e)}",
                "success": False
            })
        }

def handle_status(headers: Dict[str, str]):
    """
    Handle status requests
    """
    try:
        # Basic health check
        client = get_lobster_client()
        
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps({
                "status": "healthy",
                "version": "2.0.0",
                "environment": "aws-lambda",
                "success": True
            })
        }
        
    except Exception as e:
        return {
            "statusCode": 503,
            "headers": headers,
            "body": json.dumps({
                "status": "unhealthy",
                "error": str(e),
                "success": False
            })
        }

def handle_usage(api_key: str, user_info: Dict[str, Any], headers: Dict[str, str]):
    """
    Handle usage requests
    """
    return {
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps({
            "api_key": api_key[:8] + "...",
            "user_name": user_info["name"],
            "tier": user_info["tier"],
            "max_queries_per_hour": user_info["max_queries_per_hour"],
            "queries_used_today": 0,  # TODO: Implement usage tracking with DynamoDB
            "success": True
        })
    }

def handle_models(headers: Dict[str, str]):
    """
    Handle models list requests
    """
    return {
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps({
            "models": [
                {
                    "name": "lobster-local",
                    "description": "Full Lobster AI capabilities running on cloud infrastructure",
                    "status": "available"
                }
            ],
            "success": True
        })
    }

# For local testing
if __name__ == "__main__":
    # Test event for local development
    test_event = {
        "httpMethod": "POST",
        "path": "/query",
        "headers": {
            "Authorization": "Bearer test-enterprise-001",
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "query": "What is RNA-seq?"
        })
    }
    
    class MockContext:
        aws_request_id = "test-request-123"
    
    result = lambda_handler(test_event, MockContext())
    print("Test Result:")
    print(json.dumps(result, indent=2))
