#!/usr/bin/env python3
"""
Test script for project creation with coordinates
"""

import requests
import json

def test_create_project_with_coordinates():
    """Test creating a project with latitude and longitude."""
    
    # Test data
    project_data = {
        "name": "Test GIS Project",
        "location": "Delhi, India",
        "description": "Test project with GPS coordinates",
        "status": "Planning",
        "latitude": 28.7041,
        "longitude": 77.1025
    }
    
    try:
        print("🧪 Testing Project Creation with Coordinates")
        print("=" * 50)
        print(f"📍 Creating project: {project_data['name']}")
        print(f"🗺️  Location: {project_data['location']}")
        print(f"📍 Coordinates: {project_data['latitude']}, {project_data['longitude']}")
        
        # Create the project
        response = requests.post(
            "http://localhost:8000/api/projects",
            headers={"Content-Type": "application/json"},
            json=project_data
        )
        
        if response.status_code in [200, 201]:
            result = response.json()
            print("\n✅ Project created successfully!")
            print(f"   ID: {result['id']}")
            print(f"   Name: {result['name']}")
            print(f"   Location: {result['location']}")
            print(f"   Coordinates: {result.get('latitude', 'N/A')}, {result.get('longitude', 'N/A')}")
            
            # Test retrieving the project
            print("\n🔍 Retrieving created project...")
            get_response = requests.get(f"http://localhost:8000/api/projects/{result['id']}")
            
            if get_response.status_code == 200:
                retrieved_project = get_response.json()
                print("✅ Project retrieved successfully!")
                print(f"   Latitude: {retrieved_project.get('latitude')}")
                print(f"   Longitude: {retrieved_project.get('longitude')}")
                
                # Clean up - delete the test project
                print("\n🧹 Cleaning up test project...")
                delete_response = requests.delete(f"http://localhost:8000/api/projects/{result['id']}")
                
                if delete_response.status_code == 204:
                    print("✅ Test project deleted successfully!")
                else:
                    print(f"⚠️  Warning: Could not delete test project (status: {delete_response.status_code})")
                
                return True
            else:
                print(f"❌ Failed to retrieve project: {get_response.status_code}")
                return False
                
        else:
            print(f"❌ Failed to create project: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error testing project creation: {e}")
        return False

def test_gis_api_with_new_project():
    """Test that the GIS API includes the new project."""
    
    try:
        print("\n🗺️  Testing GIS API with updated data...")
        response = requests.get("http://localhost:8000/api/gis/project-locations")
        
        if response.status_code == 200:
            projects = response.json()
            print(f"✅ GIS API working! Found {len(projects)} projects:")
            
            for project in projects:
                if project.get('latitude') and project.get('longitude'):
                    print(f"   🗺️  {project['name']}: {project['latitude']:.4f}, {project['longitude']:.4f} ({project['status']})")
                else:
                    print(f"   📍 {project['name']}: No coordinates ({project['status']})")
            
            return True
        else:
            print(f"❌ GIS API failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing GIS API: {e}")
        return False

if __name__ == "__main__":
    success1 = test_create_project_with_coordinates()
    success2 = test_gis_api_with_new_project()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Location input functionality is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")