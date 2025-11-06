#!/usr/bin/env python3
"""
Test script to create projects with and without coordinates for display testing
"""

import requests
import json

def create_test_projects():
    """Create test projects with different coordinate scenarios."""
    
    projects = [
        {
            "name": "Mumbai Office Complex",
            "location": "Mumbai, Maharashtra",
            "description": "Commercial office building",
            "status": "In Progress",
            "latitude": 19.0760,
            "longitude": 72.8777
        },
        {
            "name": "Delhi Metro Station",
            "location": "New Delhi",
            "description": "Metro station construction",
            "status": "Planning",
            "latitude": 28.7041,
            "longitude": 77.1025
        },
        {
            "name": "Bangalore IT Park",
            "location": "Bangalore, Karnataka",
            "description": "IT park development",
            "status": "Completed",
            # No coordinates - to test display without GPS
        }
    ]
    
    created_projects = []
    
    print("🏗️  Creating Test Projects for Display Testing")
    print("=" * 50)
    
    for project_data in projects:
        try:
            print(f"\n📍 Creating: {project_data['name']}")
            if 'latitude' in project_data:
                print(f"   🗺️  With GPS: {project_data['latitude']}, {project_data['longitude']}")
            else:
                print(f"   📍 Without GPS coordinates")
            
            response = requests.post(
                "http://localhost:8000/api/projects",
                headers={"Content-Type": "application/json"},
                json=project_data
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                created_projects.append(result['id'])
                print(f"   ✅ Created with ID: {result['id']}")
            else:
                print(f"   ❌ Failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return created_projects

def test_project_list_display():
    """Test the project list API to see the display data."""
    
    try:
        print("\n📋 Testing Project List Display")
        print("=" * 30)
        
        response = requests.get("http://localhost:8000/api/projects")
        
        if response.status_code == 200:
            projects = response.json()
            print(f"✅ Found {len(projects)} projects:")
            
            for project in projects:
                print(f"\n   📋 {project['name']}")
                print(f"      Location: {project.get('location', 'N/A')}")
                print(f"      Status: {project.get('status', 'N/A')}")
                if project.get('latitude') and project.get('longitude'):
                    print(f"      GPS: {project['latitude']:.4f}, {project['longitude']:.4f} ✅")
                else:
                    print(f"      GPS: Not available ❌")
            
            return True
        else:
            print(f"❌ Failed to get projects: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def cleanup_test_projects(project_ids):
    """Clean up test projects."""
    
    print(f"\n🧹 Cleaning up {len(project_ids)} test projects...")
    
    for project_id in project_ids:
        try:
            response = requests.delete(f"http://localhost:8000/api/projects/{project_id}")
            if response.status_code == 204:
                print(f"   ✅ Deleted project {project_id}")
            else:
                print(f"   ⚠️  Could not delete project {project_id}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error deleting project {project_id}: {e}")

if __name__ == "__main__":
    print("🧪 Project Display Testing")
    print("=" * 50)
    
    # Create test projects
    created_ids = create_test_projects()
    
    if created_ids:
        # Test display
        test_project_list_display()
        
        # Ask user if they want to keep the test data
        print(f"\n❓ Keep test projects for manual testing? (y/N): ", end="")
        try:
            keep = input().lower().strip()
            if keep != 'y':
                cleanup_test_projects(created_ids)
            else:
                print("✅ Test projects kept for manual testing.")
                print("   Visit http://localhost:8000/projects to see the display.")
        except KeyboardInterrupt:
            print("\n🧹 Cleaning up...")
            cleanup_test_projects(created_ids)
    else:
        print("❌ No test projects were created successfully.")