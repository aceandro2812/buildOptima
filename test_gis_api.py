#!/usr/bin/env python3
"""
Test script for the GIS API endpoint
"""

import requests
import json

def test_project_locations_api():
    """Test the project locations API endpoint."""
    
    try:
        # Test the API endpoint
        response = requests.get("http://localhost:8000/api/gis/project-locations")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API endpoint working!")
            print(f"📍 Found {len(data)} projects")
            
            for project in data:
                if project.get('latitude') and project.get('longitude'):
                    print(f"   🗺️  {project['name']}: {project['latitude']:.4f}, {project['longitude']:.4f} ({project['status']})")
                else:
                    print(f"   📍 {project['name']}: No coordinates ({project['status']})")
            
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing GIS API Integration")
    print("=" * 40)
    
    success = test_project_locations_api()
    
    if success:
        print("\n🎉 GIS API integration test passed!")
        print("You can now view the dashboard at: http://localhost:8000")
    else:
        print("\n❌ GIS API integration test failed!")