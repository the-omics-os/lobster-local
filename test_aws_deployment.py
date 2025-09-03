#!/usr/bin/env python3
"""
AWS Deployment Test Script for Lobster Cloud
Comprehensive testing of the deployed Lambda function and API Gateway
"""

import json
import os
import sys
import time
import requests
from typing import Dict, Any, Optional
import argparse
from dataclasses import dataclass

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color

@dataclass
class TestConfig:
    """Configuration for testing"""
    endpoint: str
    api_keys: Dict[str, str]
    timeout: int = 30
    verbose: bool = False

class LobsterCloudTester:
    """Comprehensive tester for Lobster Cloud deployment"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.session = requests.Session()
        self.test_results = []
        
        # Set up session with default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "Lobster-Cloud-Tester/1.0"
        })
    
    def log(self, message: str, color: str = Colors.WHITE):
        """Log a message with color"""
        print(f"{color}{message}{Colors.NC}")
    
    def log_success(self, message: str):
        """Log a success message"""
        self.log(f"✅ {message}", Colors.GREEN)
    
    def log_error(self, message: str):
        """Log an error message"""
        self.log(f"❌ {message}", Colors.RED)
    
    def log_warning(self, message: str):
        """Log a warning message"""
        self.log(f"⚠️  {message}", Colors.YELLOW)
    
    def log_info(self, message: str):
        """Log an info message"""
        self.log(f"ℹ️  {message}", Colors.BLUE)
    
    def record_test_result(self, test_name: str, success: bool, message: str, duration: float = 0):
        """Record test result"""
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "duration": duration
        })
    
    def make_request(self, method: str, path: str, api_key: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a request to the API"""
        url = f"{self.config.endpoint.rstrip('/')}/{path.lstrip('/')}"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            start_time = time.time()
            
            if method.upper() == "POST":
                response = self.session.post(
                    url, 
                    json=data or {}, 
                    headers=headers, 
                    timeout=self.config.timeout
                )
            elif method.upper() == "GET":
                response = self.session.get(
                    url, 
                    headers=headers, 
                    timeout=self.config.timeout
                )
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            duration = time.time() - start_time
            
            # Try to parse JSON response
            try:
                result = response.json()
            except json.JSONDecodeError:
                result = {"raw_response": response.text}
            
            result.update({
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "duration": duration
            })
            
            return result
            
        except requests.exceptions.Timeout:
            return {
                "error": "Request timeout",
                "status_code": 408,
                "duration": self.config.timeout
            }
        except requests.exceptions.ConnectionError:
            return {
                "error": "Connection error",
                "status_code": 0,
                "duration": 0
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {e}",
                "status_code": 0,
                "duration": 0
            }
    
    def test_status_endpoint(self) -> bool:
        """Test the status endpoint"""
        self.log("Testing status endpoint...", Colors.CYAN)
        
        # Test with valid API key
        api_key = list(self.config.api_keys.values())[0]
        result = self.make_request("POST", "/status", api_key)
        
        success = (
            result.get("status_code") == 200 and
            result.get("status") == "healthy" and
            result.get("success", False)
        )
        
        if success:
            self.log_success(f"Status endpoint working (Response time: {result.get('duration', 0):.2f}s)")
            self.record_test_result("status_endpoint", True, "Status endpoint healthy", result.get('duration', 0))
        else:
            error_msg = result.get("error", f"Status: {result.get('status_code')}")
            self.log_error(f"Status endpoint failed: {error_msg}")
            self.record_test_result("status_endpoint", False, error_msg)
        
        if self.config.verbose:
            self.log(f"Full response: {json.dumps(result, indent=2)}", Colors.WHITE)
        
        return success
    
    def test_api_key_validation(self) -> bool:
        """Test API key validation"""
        self.log("Testing API key validation...", Colors.CYAN)
        
        # Test invalid API key
        invalid_result = self.make_request("POST", "/status", "invalid-key-123")
        
        invalid_success = invalid_result.get("status_code") == 401
        
        if invalid_success:
            self.log_success("Invalid API key correctly rejected")
        else:
            self.log_error(f"Invalid API key not rejected properly: {invalid_result.get('status_code')}")
        
        # Test each valid API key
        valid_success = True
        for key_name, api_key in self.config.api_keys.items():
            result = self.make_request("POST", "/status", api_key)
            if result.get("status_code") != 200:
                self.log_error(f"Valid API key '{key_name}' rejected: {result.get('status_code')}")
                valid_success = False
            else:
                self.log_success(f"API key '{key_name}' accepted")
        
        success = invalid_success and valid_success
        self.record_test_result("api_key_validation", success, "API key validation working" if success else "API key validation failed")
        
        return success
    
    def test_query_processing(self) -> bool:
        """Test query processing functionality"""
        self.log("Testing query processing...", Colors.CYAN)
        
        # Test queries with different complexity
        test_queries = [
            {
                "name": "Simple query",
                "query": "What is RNA-seq?",
                "expected_duration": 30
            },
            {
                "name": "Bioinformatics query",
                "query": "Explain the difference between RNA-seq and ChIP-seq",
                "expected_duration": 45
            },
            {
                "name": "Technical query",
                "query": "How do I perform quality control on sequencing data?",
                "expected_duration": 60
            }
        ]
        
        api_key = list(self.config.api_keys.values())[0]
        all_success = True
        
        for test_case in test_queries:
            self.log(f"Testing: {test_case['name']}")
            
            data = {
                "query": test_case["query"],
                "options": {}
            }
            
            result = self.make_request("POST", "/query", api_key, data)
            
            success = (
                result.get("status_code") == 200 and
                result.get("success", False) and
                "response" in result
            )
            
            duration = result.get("duration", 0)
            
            if success:
                self.log_success(f"{test_case['name']} processed successfully ({duration:.2f}s)")
                if self.config.verbose:
                    response_preview = result.get("response", "")[:100] + "..."
                    self.log(f"Response preview: {response_preview}", Colors.WHITE)
            else:
                error_msg = result.get("error", f"Status: {result.get('status_code')}")
                self.log_error(f"{test_case['name']} failed: {error_msg}")
                all_success = False
            
            # Check if duration is reasonable
            if duration > test_case["expected_duration"]:
                self.log_warning(f"Query took longer than expected: {duration:.2f}s > {test_case['expected_duration']}s")
            
            self.record_test_result(
                f"query_{test_case['name'].lower().replace(' ', '_')}", 
                success, 
                f"Processed in {duration:.2f}s" if success else error_msg,
                duration
            )
        
        return all_success
    
    def test_usage_endpoint(self) -> bool:
        """Test usage information endpoint"""
        self.log("Testing usage endpoint...", Colors.CYAN)
        
        api_key = list(self.config.api_keys.values())[0]
        result = self.make_request("POST", "/usage", api_key)
        
        success = (
            result.get("status_code") == 200 and
            result.get("success", False) and
            "tier" in result
        )
        
        if success:
            self.log_success(f"Usage endpoint working - Tier: {result.get('tier')}")
            if self.config.verbose:
                self.log(f"Usage info: {json.dumps({k:v for k,v in result.items() if k not in ['headers']}, indent=2)}", Colors.WHITE)
        else:
            error_msg = result.get("error", f"Status: {result.get('status_code')}")
            self.log_error(f"Usage endpoint failed: {error_msg}")
        
        self.record_test_result("usage_endpoint", success, "Usage endpoint working" if success else error_msg)
        
        return success
    
    def test_models_endpoint(self) -> bool:
        """Test models listing endpoint"""
        self.log("Testing models endpoint...", Colors.CYAN)
        
        api_key = list(self.config.api_keys.values())[0]
        result = self.make_request("POST", "/models", api_key)
        
        success = (
            result.get("status_code") == 200 and
            result.get("success", False) and
            "models" in result
        )
        
        if success:
            models_count = len(result.get("models", []))
            self.log_success(f"Models endpoint working - {models_count} models available")
        else:
            error_msg = result.get("error", f"Status: {result.get('status_code')}")
            self.log_error(f"Models endpoint failed: {error_msg}")
        
        self.record_test_result("models_endpoint", success, "Models endpoint working" if success else error_msg)
        
        return success
    
    def test_cors_support(self) -> bool:
        """Test CORS support for browser compatibility"""
        self.log("Testing CORS support...", Colors.CYAN)
        
        # Test OPTIONS preflight request
        url = f"{self.config.endpoint}/query"
        
        try:
            response = requests.options(
                url,
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type, Authorization"
                },
                timeout=10
            )
            
            cors_headers = {
                "Access-Control-Allow-Origin": response.headers.get("Access-Control-Allow-Origin"),
                "Access-Control-Allow-Methods": response.headers.get("Access-Control-Allow-Methods"),
                "Access-Control-Allow-Headers": response.headers.get("Access-Control-Allow-Headers")
            }
            
            success = (
                response.status_code == 200 and
                cors_headers["Access-Control-Allow-Origin"] is not None and
                cors_headers["Access-Control-Allow-Methods"] is not None
            )
            
            if success:
                self.log_success("CORS support working")
                if self.config.verbose:
                    self.log(f"CORS headers: {json.dumps(cors_headers, indent=2)}", Colors.WHITE)
            else:
                self.log_error(f"CORS support failed: Status {response.status_code}")
            
        except Exception as e:
            success = False
            self.log_error(f"CORS test failed: {e}")
        
        self.record_test_result("cors_support", success, "CORS working" if success else "CORS failed")
        
        return success
    
    def test_error_handling(self) -> bool:
        """Test error handling scenarios"""
        self.log("Testing error handling...", Colors.CYAN)
        
        api_key = list(self.config.api_keys.values())[0]
        
        # Test empty query
        empty_result = self.make_request("POST", "/query", api_key, {"query": ""})
        empty_success = empty_result.get("status_code") == 400
        
        # Test malformed JSON
        try:
            url = f"{self.config.endpoint}/query"
            malformed_response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                data="invalid json",
                timeout=10
            )
            malformed_success = malformed_response.status_code == 400
        except:
            malformed_success = False
        
        # Test invalid endpoint
        invalid_result = self.make_request("POST", "/invalid-endpoint", api_key)
        invalid_success = invalid_result.get("status_code") == 404
        
        success = empty_success and malformed_success and invalid_success
        
        if success:
            self.log_success("Error handling working correctly")
        else:
            self.log_error("Error handling has issues")
            if not empty_success:
                self.log_error("Empty query not handled properly")
            if not malformed_success:
                self.log_error("Malformed JSON not handled properly")
            if not invalid_success:
                self.log_error("Invalid endpoint not handled properly")
        
        self.record_test_result("error_handling", success, "Error handling working" if success else "Error handling failed")
        
        return success
    
    def test_performance(self) -> bool:
        """Test performance characteristics"""
        self.log("Testing performance...", Colors.CYAN)
        
        api_key = list(self.config.api_keys.values())[0]
        
        # Test multiple concurrent requests (simulated)
        query_times = []
        for i in range(3):
            result = self.make_request("POST", "/status", api_key)
            if result.get("status_code") == 200:
                query_times.append(result.get("duration", 0))
        
        if query_times:
            avg_time = sum(query_times) / len(query_times)
            max_time = max(query_times)
            
            # Performance thresholds
            success = avg_time < 10.0 and max_time < 30.0
            
            if success:
                self.log_success(f"Performance acceptable - Avg: {avg_time:.2f}s, Max: {max_time:.2f}s")
            else:
                self.log_warning(f"Performance may be slow - Avg: {avg_time:.2f}s, Max: {max_time:.2f}s")
            
            self.record_test_result(
                "performance", 
                success, 
                f"Avg: {avg_time:.2f}s, Max: {max_time:.2f}s",
                avg_time
            )
        else:
            success = False
            self.log_error("Could not measure performance")
            self.record_test_result("performance", False, "Could not measure performance")
        
        return success
    
    def run_all_tests(self) -> bool:
        """Run all tests"""
        self.log("Starting Lobster Cloud deployment tests...", Colors.MAGENTA)
        self.log(f"Testing endpoint: {self.config.endpoint}", Colors.BLUE)
        self.log(f"API keys: {len(self.config.api_keys)} configured", Colors.BLUE)
        print()
        
        tests = [
            ("Status Endpoint", self.test_status_endpoint),
            ("API Key Validation", self.test_api_key_validation),
            ("Usage Endpoint", self.test_usage_endpoint),
            ("Models Endpoint", self.test_models_endpoint),
            ("CORS Support", self.test_cors_support),
            ("Error Handling", self.test_error_handling),
            ("Performance", self.test_performance),
            ("Query Processing", self.test_query_processing),  # Run this last as it's slowest
        ]
        
        results = []
        start_time = time.time()
        
        for test_name, test_func in tests:
            self.log(f"\n--- {test_name} ---", Colors.CYAN)
            try:
                result = test_func()
                results.append(result)
            except Exception as e:
                self.log_error(f"Test {test_name} crashed: {e}")
                results.append(False)
                self.record_test_result(test_name.lower().replace(' ', '_'), False, f"Test crashed: {e}")
        
        total_time = time.time() - start_time
        
        # Print summary
        print("\n" + "="*60)
        self.log("TEST SUMMARY", Colors.MAGENTA)
        print("="*60)
        
        passed = sum(results)
        total = len(results)
        
        for result in self.test_results:
            status_color = Colors.GREEN if result["success"] else Colors.RED
            status_symbol = "✅" if result["success"] else "❌"
            duration_info = f" ({result['duration']:.2f}s)" if result["duration"] > 0 else ""
            
            self.log(f"{status_symbol} {result['test']}: {result['message']}{duration_info}", status_color)
        
        print("-" * 60)
        
        if passed == total:
            self.log_success(f"ALL TESTS PASSED ({passed}/{total}) in {total_time:.2f}s")
            self.log_success("🎉 Lobster Cloud deployment is ready for business validation!")
            return True
        else:
            self.log_error(f"SOME TESTS FAILED ({passed}/{total}) in {total_time:.2f}s")
            self.log_warning("Check the failures above and fix before proceeding to business validation")
            return False

def main():
    parser = argparse.ArgumentParser(description="Test Lobster Cloud AWS deployment")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("LOBSTER_ENDPOINT", "https://your-api-id.execute-api.us-east-1.amazonaws.com/prod"),
        help="API Gateway endpoint URL"
    )
    parser.add_argument(
        "--api-key",
        action="append",
        help="API key to test with (can be used multiple times)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds"
    )
    
    args = parser.parse_args()
    
    # Configure API keys
    api_keys = {}
    if args.api_key:
        for i, key in enumerate(args.api_key):
            api_keys[f"key_{i+1}"] = key
    else:
        # Use environment variables or defaults
        default_keys = {
            "enterprise_1": os.getenv("LOBSTER_CLOUD_KEY", "test-enterprise-001"),
            "enterprise_2": os.getenv("LOBSTER_CLOUD_KEY_2", "test-enterprise-002"),
            "demo": os.getenv("LOBSTER_DEMO_KEY", "demo-user-001")
        }
        api_keys = {k: v for k, v in default_keys.items() if v}
    
    if not api_keys:
        print(f"{Colors.RED}❌ No API keys configured. Set LOBSTER_CLOUD_KEY or use --api-key{Colors.NC}")
        sys.exit(1)
    
    # Validate endpoint
    if "your-api-id" in args.endpoint:
        print(f"{Colors.YELLOW}⚠️  Warning: Using default endpoint. Set LOBSTER_ENDPOINT or use --endpoint{Colors.NC}")
        print(f"{Colors.YELLOW}   Current endpoint: {args.endpoint}{Colors.NC}")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting. Configure your endpoint first.")
            sys.exit(1)
    
    # Create test configuration
    config = TestConfig(
        endpoint=args.endpoint,
        api_keys=api_keys,
        timeout=args.timeout,
        verbose=args.verbose
    )
    
    # Run tests
    tester = LobsterCloudTester(config)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
