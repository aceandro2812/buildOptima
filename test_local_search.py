#!/usr/bin/env python3
"""
Test script for the local search tool.
Run this to verify the search functionality works correctly.
"""

import sys
import os

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from tools.local_search import LocalSearchTool, search_material_prices

def test_basic_search():
    """Test basic search functionality."""
    print("=" * 60)
    print("Testing Basic Search Functionality")
    print("=" * 60)
    
    search_tool = LocalSearchTool(max_results=2, delay_range=(0.5, 1))
    
    # Test Google search
    print("\n1. Testing Google search...")
    try:
        results = search_tool.search("cement price Mumbai", engine='google')
        if results:
            print(f"✓ Found {len(results)} results from Google")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result.title[:50]}...")
                print(f"      URL: {result.url}")
        else:
            print("✗ No results from Google")
    except Exception as e:
        print(f"✗ Google search failed: {e}")
    
    # Test Bing search
    print("\n2. Testing Bing search...")
    try:
        results = search_tool.search("steel price India", engine='bing')
        if results:
            print(f"✓ Found {len(results)} results from Bing")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result.title[:50]}...")
                print(f"      URL: {result.url}")
        else:
            print("✗ No results from Bing")
    except Exception as e:
        print(f"✗ Bing search failed: {e}")

def test_multi_engine_search():
    """Test multi-engine search functionality."""
    print("\n" + "=" * 60)
    print("Testing Multi-Engine Search")
    print("=" * 60)
    
    search_tool = LocalSearchTool(max_results=3, delay_range=(0.5, 1))
    
    try:
        results = search_tool.multi_engine_search("concrete price rate India")
        if results:
            print(f"✓ Found {len(results)} combined results")
            for i, result in enumerate(results, 1):
                print(f"   {i}. [{result.source.upper()}] {result.title[:40]}...")
                print(f"      URL: {result.url}")
        else:
            print("✗ No combined results found")
    except Exception as e:
        print(f"✗ Multi-engine search failed: {e}")

def test_construction_material_search():
    """Test specialized construction material search."""
    print("\n" + "=" * 60)
    print("Testing Construction Material Search")
    print("=" * 60)
    
    materials = ["cement", "steel bars", "bricks"]
    
    for material in materials:
        print(f"\nTesting search for: {material}")
        try:
            result = search_material_prices(material, "Mumbai")
            if "No search results" not in result and "Error occurred" not in result:
                print(f"✓ Successfully found pricing info for {material}")
                # Show first few lines of result
                lines = result.split('\n')[:5]
                for line in lines:
                    if line.strip():
                        print(f"   {line[:80]}...")
            else:
                print(f"✗ No pricing info found for {material}")
        except Exception as e:
            print(f"✗ Search failed for {material}: {e}")

def main():
    """Run all tests."""
    print("Local Search Tool Test Suite")
    print("This will test the web crawling functionality")
    print("Note: Tests involve actual web requests with delays")
    
    try:
        test_basic_search()
        test_multi_engine_search()
        test_construction_material_search()
        
        print("\n" + "=" * 60)
        print("Test Suite Completed!")
        print("=" * 60)
        print("\nIf you see ✓ marks above, the search tool is working correctly.")
        print("The tool can now replace DuckDuckGo in your agents.")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")

if __name__ == "__main__":
    main()