"""
Test for LangChain's ChatBedrockConverse with concurrent requests.

This script tests sending multiple concurrent requests to AWS Bedrock
to check quota limits by attempting to send 20 requests simultaneously.
"""

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import concurrent.futures
import time
from datetime import datetime
import traceback


def single_request(llm, request_id):
    """Execute a single request to Bedrock."""
    start_time = time.time()
    result = {"id": request_id, "status": "success", "error": None, "duration": 0}
    
    try:
        print(f"[{request_id}] Starting request...")
        # Using LangChain message objects instead of a dictionary
        messages = [
            HumanMessage(content=f"Short answer: what is the number {request_id}?")
        ]
        
        # Collect all chunks into a response
        response = ""
        for chunk in llm.invoke(messages):
            response += str(chunk)
            
        end_time = time.time()
        result["duration"] = end_time - start_time
        result["response_length"] = len(response)
        print(f"[{request_id}] ✓ Request completed in {result['duration']:.2f}s")
        
    except Exception as e:
        end_time = time.time()
        result["status"] = "error"
        result["error"] = str(e)
        result["duration"] = end_time - start_time
        print(f"[{request_id}] ✗ Request failed: {str(e)}")
        
    return result


def test_chat_bedrock_converse():
    """Test ChatBedrockConverse with 20 concurrent requests."""
    print(f"Starting concurrent requests test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Attempting to send 20 concurrent requests to AWS Bedrock...")
    
    # Load environment variables (AWS credentials)
    load_dotenv()
    
    # Track results
    results = {
        "successful_requests": 0,
        "failed_requests": 0,
        "total_duration": 0,
        "requests": []
    }
    
    try:
        # Create the LLM with minimal parameters
        llm = ChatBedrockConverse(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            temperature=1,
            region_name="us-east-1"
        )
        print("✓ Successfully created ChatBedrockConverse instance")

        # Number of concurrent requests
        num_requests = 20
        
        # Start timing
        overall_start = time.time()
        
        # Use ThreadPoolExecutor for concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            # Submit all requests
            future_to_id = {
                executor.submit(single_request, llm, i): i 
                for i in range(1, num_requests + 1)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_id):
                request_id = future_to_id[future]
                try:
                    result = future.result()
                    results["requests"].append(result)
                    
                    if result["status"] == "success":
                        results["successful_requests"] += 1
                    else:
                        results["failed_requests"] += 1
                        
                except Exception as e:
                    print(f"[{request_id}] Request raised exception: {str(e)}")
                    results["requests"].append({
                        "id": request_id,
                        "status": "exception",
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
                    results["failed_requests"] += 1
        
        # Calculate overall duration
        overall_end = time.time()
        results["total_duration"] = overall_end - overall_start
        
        # Print summary
        print("\n--- TEST SUMMARY ---")
        print(f"Total requests attempted: {num_requests}")
        print(f"Successful requests: {results['successful_requests']}")
        print(f"Failed requests: {results['failed_requests']}")
        print(f"Total duration: {results['total_duration']:.2f} seconds")
        
        # Print details of any failed requests
        if results["failed_requests"] > 0:
            print("\nFailed requests:")
            for req in results["requests"]:
                if req["status"] != "success":
                    print(f"  Request {req['id']}: {req['error']}")
        
        # Check if we hit the quota limit
        if any("quota" in str(req.get("error", "")).lower() for req in results["requests"]):
            print("\n⚠️ QUOTA LIMIT DETECTED: Some requests failed due to quota limitations")
        
        # Print success rate
        success_rate = (results["successful_requests"] / num_requests) * 100
        print(f"\nSuccess rate: {success_rate:.2f}%")
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nPossible issues:")
        print("1. Check AWS credentials are correctly configured")
        print("2. Verify the model ID is correct and available in your AWS region")
        print("3. Check that you have proper permissions to access the model")


if __name__ == "__main__":
    test_chat_bedrock_converse()
