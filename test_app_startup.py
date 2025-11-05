#!/usr/bin/env python3
"""
Test script to verify the application can start without errors
"""

import sys
import os
import asyncio
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_app_startup():
    """Test if the FastAPI app can start without errors"""
    try:
        # Import the main app
        from main import app
        print("✅ Main app imported successfully")
        
        # Test if we can access the routes
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test dashboard route
        response = client.get("/dashboard")
        if response.status_code == 200:
            print("✅ Dashboard route accessible")
        else:
            print(f"❌ Dashboard route failed: {response.status_code}")
            
        # Test if serialization works by checking if pages load
        try:
            response = client.get("/costs")
            if response.status_code == 200:
                print("✅ Costs page loads successfully (serialization working)")
            else:
                print(f"❌ Costs page failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Costs page error: {e}")
            
        try:
            response = client.get("/consumption")
            if response.status_code == 200:
                print("✅ Consumption page loads successfully")
            else:
                print(f"❌ Consumption page failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Consumption page error: {e}")
            
        try:
            response = client.get("/waste")
            if response.status_code == 200:
                print("✅ Waste page loads successfully")
            else:
                print(f"❌ Waste page failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Waste page error: {e}")
            
        print("\n🎉 Application startup test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Application startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_app_startup())