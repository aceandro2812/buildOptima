#!/usr/bin/env python3
"""
Test script to verify backend serialization fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import serialize_cost, serialize_consumption, serialize_waste
from models import Cost, Consumption, Waste, Inventory, Supplier, Project
from datetime import datetime

def test_serialization():
    """Test the serialization functions"""
    
    # Create mock objects
    class MockMaterial:
        def __init__(self):
            self.material_name = "Test Material"
            self.unit = "kg"
    
    class MockSupplier:
        def __init__(self):
            self.name = "Test Supplier"
    
    class MockProject:
        def __init__(self):
            self.name = "Test Project"
    
    class MockCost:
        def __init__(self):
            self.id = 1
            self.material_id = 1
            self.supplier_id = 1
            self.unit_price = 10.50
            self.quantity_purchased = 100.0
            self.total_cost = 1050.0
            self.date_recorded = datetime.now()
            self.notes = "Test cost"
            self.material = MockMaterial()
            self.supplier = MockSupplier()
    
    class MockConsumption:
        def __init__(self):
            self.id = 1
            self.material_id = 1
            self.project_id = 1
            self.quantity_used = 50.0
            self.date_used = datetime.now()
            self.notes = "Test consumption"
            self.material = MockMaterial()
            self.project_rel = MockProject()
    
    class MockWaste:
        def __init__(self):
            self.id = 1
            self.material_id = 1
            self.project_id = 1
            self.quantity_wasted = 5.0
            self.date_recorded = datetime.now()
            self.reason = "Damaged"
            self.preventive_measures = "Better storage"
            self.material = MockMaterial()
            self.project_rel = MockProject()
    
    # Test serialization
    try:
        cost = MockCost()
        cost_dict = serialize_cost(cost)
        print("✅ Cost serialization successful")
        print(f"   Sample: {cost_dict['material']['material_name']} - ${cost_dict['total_cost']}")
        
        consumption = MockConsumption()
        consumption_dict = serialize_consumption(consumption)
        print("✅ Consumption serialization successful")
        print(f"   Sample: {consumption_dict['material']['material_name']} - {consumption_dict['quantity_used']} units")
        
        waste = MockWaste()
        waste_dict = serialize_waste(waste)
        print("✅ Waste serialization successful")
        print(f"   Sample: {waste_dict['material_name']} - {waste_dict['quantity_wasted']} units")
        
        print("\n🎉 All serialization functions working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        return False

if __name__ == "__main__":
    test_serialization()