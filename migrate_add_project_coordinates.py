#!/usr/bin/env python3
"""
Database migration script to add latitude and longitude columns to projects table.
This adds basic GIS functionality to BuildOptima.
"""

import sqlite3
import os
import sys

def add_coordinates_to_projects():
    """Add latitude and longitude columns to the projects table."""
    
    # Database file path
    db_path = "construction_materials.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database file {db_path} not found!")
        print("Please make sure you're running this from the correct directory.")
        return False
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(projects)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'latitude' in columns and 'longitude' in columns:
            print("✅ Latitude and longitude columns already exist in projects table.")
            conn.close()
            return True
        
        print("🔄 Adding latitude and longitude columns to projects table...")
        
        # Add latitude column
        if 'latitude' not in columns:
            cursor.execute("ALTER TABLE projects ADD COLUMN latitude REAL")
            print("   ✅ Added latitude column")
        
        # Add longitude column  
        if 'longitude' not in columns:
            cursor.execute("ALTER TABLE projects ADD COLUMN longitude REAL")
            print("   ✅ Added longitude column")
        
        # Commit changes
        conn.commit()
        
        # Verify the changes
        cursor.execute("PRAGMA table_info(projects)")
        updated_columns = [column[1] for column in cursor.fetchall()]
        
        if 'latitude' in updated_columns and 'longitude' in updated_columns:
            print("✅ Migration completed successfully!")
            print("   Projects table now has latitude and longitude columns for GIS functionality.")
        else:
            print("❌ Migration verification failed!")
            return False
            
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        if conn:
            conn.close()
    
    return True

def add_sample_coordinates():
    """Add sample coordinates to existing projects for testing."""
    
    # Sample coordinates for major Indian cities (for construction projects)
    sample_locations = [
        {"name": "Mumbai", "lat": 19.0760, "lng": 72.8777},
        {"name": "Delhi", "lat": 28.7041, "lng": 77.1025},
        {"name": "Bangalore", "lat": 12.9716, "lng": 77.5946},
        {"name": "Chennai", "lat": 13.0827, "lng": 80.2707},
        {"name": "Hyderabad", "lat": 17.3850, "lng": 78.4867},
        {"name": "Pune", "lat": 18.5204, "lng": 73.8567},
    ]
    
    try:
        conn = sqlite3.connect("construction_materials.db")
        cursor = conn.cursor()
        
        # Get existing projects without coordinates
        cursor.execute("SELECT id, name FROM projects WHERE latitude IS NULL OR longitude IS NULL")
        projects = cursor.fetchall()
        
        if not projects:
            print("ℹ️  No projects need sample coordinates.")
            return True
        
        print(f"🔄 Adding sample coordinates to {len(projects)} projects...")
        
        # Assign sample coordinates to projects
        for i, (project_id, project_name) in enumerate(projects):
            # Use modulo to cycle through sample locations
            location = sample_locations[i % len(sample_locations)]
            
            # Add some random offset to make locations unique
            lat_offset = (i * 0.01) - 0.05  # Small offset
            lng_offset = (i * 0.01) - 0.05
            
            final_lat = location["lat"] + lat_offset
            final_lng = location["lng"] + lng_offset
            
            cursor.execute(
                "UPDATE projects SET latitude = ?, longitude = ? WHERE id = ?",
                (final_lat, final_lng, project_id)
            )
            
            print(f"   ✅ {project_name}: {final_lat:.4f}, {final_lng:.4f}")
        
        conn.commit()
        print("✅ Sample coordinates added successfully!")
        
    except sqlite3.Error as e:
        print(f"❌ Database error while adding sample coordinates: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error while adding sample coordinates: {e}")
        return False
    finally:
        if conn:
            conn.close()
    
    return True

if __name__ == "__main__":
    print("🗺️  BuildOptima GIS Migration")
    print("=" * 40)
    
    # Step 1: Add columns
    if not add_coordinates_to_projects():
        sys.exit(1)
    
    # Step 2: Ask if user wants sample data
    if len(sys.argv) > 1 and sys.argv[1] == "--sample-data":
        print("\n" + "=" * 40)
        add_sample_coordinates()
    else:
        print("\nℹ️  To add sample coordinates to existing projects, run:")
        print("   python migrate_add_project_coordinates.py --sample-data")
    
    print("\n🎉 GIS integration ready!")
    print("   You can now add latitude/longitude to projects and display them on maps.")