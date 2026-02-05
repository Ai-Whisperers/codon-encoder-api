#!/usr/bin/env python3
"""
Health check script for Codon Encoder API.
Can be used for monitoring, testing deployment, or CI/CD pipelines.
"""

import sys
import time
import urllib.request
import urllib.error
import argparse
import json
from typing import Optional


def check_api_health(base_url: str, timeout: int = 30, retries: int = 3) -> bool:
    """
    Check if the API is healthy and responding.
    
    Args:
        base_url: API base URL (e.g., http://localhost:8765)
        timeout: Timeout for each request in seconds
        retries: Number of retry attempts
        
    Returns:
        True if API is healthy, False otherwise
    """
    metadata_url = f"{base_url}/api/metadata"
    
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1}/{retries}: Checking {metadata_url}")
            
            # Make request with timeout
            req = urllib.request.Request(metadata_url)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.getcode() == 200:
                    data = json.loads(response.read().decode())
                    print(f"✓ API is healthy")
                    print(f"  Status: {response.getcode()}")
                    print(f"  Model version: {data.get('version', 'unknown')}")
                    print(f"  Embed dim: {data.get('embed_dim', 'unknown')}")
                    print(f"  Clusters: {data.get('num_clusters', 'unknown')}")
                    return True
                else:
                    print(f"✗ Unexpected status code: {response.getcode()}")
                    
        except urllib.error.HTTPError as e:
            print(f"✗ HTTP error: {e.code} - {e.reason}")
        except urllib.error.URLError as e:
            print(f"✗ URL error: {e.reason}")
        except ConnectionRefusedError:
            print(f"✗ Connection refused - is the server running?")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            
        if attempt < retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"  Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    return False


def test_basic_endpoint(base_url: str) -> bool:
    """Test basic API functionality with a simple encode request."""
    encode_url = f"{base_url}/api/encode"
    
    try:
        # Simple test sequence (ATG = start codon)
        data = json.dumps({"sequence": "ATGCCCGGGTTT"}).encode('utf-8')
        
        req = urllib.request.Request(
            encode_url, 
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.getcode() == 200:
                result = json.loads(response.read().decode())
                if isinstance(result, list) and len(result) > 0:
                    codon_data = result[0]
                    print(f"✓ Encode endpoint working")
                    print(f"  First codon: {codon_data.get('codon', 'unknown')}")
                    print(f"  Amino acid: {codon_data.get('amino_acid', 'unknown')}")
                    return True
                else:
                    print("✗ Unexpected response format")
                    return False
            else:
                print(f"✗ Encode test failed: {response.getcode()}")
                return False
                
    except Exception as e:
        print(f"✗ Encode test error: {e}")
        return False


def main():
    """Main health check function."""
    parser = argparse.ArgumentParser(description="Health check for Codon Encoder API")
    parser.add_argument(
        "--url", 
        default="http://localhost:8765", 
        help="API base URL (default: http://localhost:8765)"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=30, 
        help="Request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--retries", 
        type=int, 
        default=3, 
        help="Number of retry attempts (default: 3)"
    )
    parser.add_argument(
        "--test-encode",
        action="store_true",
        help="Also test the encode endpoint"
    )
    parser.add_argument(
        "--wait",
        type=int,
        help="Wait for this many seconds for the API to become available"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Codon Encoder API Health Check")
    print("=" * 50)
    print(f"URL: {args.url}")
    print(f"Timeout: {args.timeout}s")
    print(f"Retries: {args.retries}")
    print()
    
    # Optional: wait for API to become available
    if args.wait:
        print(f"Waiting up to {args.wait}s for API to become available...")
        start_time = time.time()
        while time.time() - start_time < args.wait:
            if check_api_health(args.url, timeout=5, retries=1):
                break
            time.sleep(2)
        else:
            print(f"API not available after {args.wait}s")
            return 1
        print()
    
    # Check basic health
    healthy = check_api_health(args.url, args.timeout, args.retries)
    
    if not healthy:
        print("\n💀 API health check failed")
        return 1
    
    # Optional: test encode endpoint
    if args.test_encode:
        print()
        encode_ok = test_basic_endpoint(args.url)
        if not encode_ok:
            print("\n💀 Encode endpoint test failed")
            return 1
    
    print("\n🎉 All health checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())