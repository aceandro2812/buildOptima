from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional

from models import Inventory, Supplier, Consumption, Cost, Waste, Alert
from schemas import *
    
# INVENTORY CRUD OPERATIONS
def get_inventory(db: Session) -> List[Inventory]:
    return db.query(Inventory).all()

def create_inventory(db: Session, item: InventoryCreate) -> Inventory:
    db_item = Inventory(**item.model_dump())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def update_inventory(db: Session, item_id: int, quantity: float) -> Inventory:
    db_item = db.query(Inventory).filter(Inventory.id == item_id).first()
    if db_item:
        db_item.quantity = quantity
        db_item.last_updated = datetime.utcnow()
        db.commit()
        db.refresh(db_item)
        # Create a low stock alert if needed
        if quantity <= db_item.reorder_point:
            create_alert(db, AlertCreate(
                material_id=item_id,
                alert_type="low_stock",
                message=f"Material {db_item.material_name} is below reorder point"
            ))
    return db_item

def update_inventory_item(db: Session, item_id: int, item: InventoryCreate) -> Optional[Inventory]:
    db_item = db.query(Inventory).filter(Inventory.id == item_id).first()
    if db_item:
        for key, value in item.model_dump().items():
            setattr(db_item, key, value)
        db_item.last_updated = datetime.utcnow()
        db.commit()
        db.refresh(db_item)
        return db_item
    return None

def get_inventory_item(db: Session, item_id: int) -> Optional[Inventory]:
    return db.query(Inventory).filter(Inventory.id == item_id).first()

def delete_inventory_item(db: Session, item_id: int) -> bool:
    db_item = db.query(Inventory).filter(Inventory.id == item_id).first()
    if db_item:
        db.delete(db_item)
        db.commit()
        return True
    return False

# SUPPLIER CRUD OPERATIONS
def get_suppliers(db: Session) -> List[Supplier]:
    return db.query(Supplier).all()

def create_supplier_db(db: Session, supplier: SupplierCreate) -> Supplier:
    db_supplier = Supplier(**supplier.model_dump())
    db.add(db_supplier)
    db.commit()
    db.refresh(db_supplier)
    return db_supplier

def get_supplier_by_id(db: Session, supplier_id: int) -> Optional[Supplier]:
    return db.query(Supplier).filter(Supplier.id == supplier_id).first()

def update_supplier_db(db: Session, supplier_id: int, supplier: SupplierCreate) -> Optional[Supplier]:
    db_supplier = db.query(Supplier).filter(Supplier.id == supplier_id).first()
    if db_supplier:
        for key, value in supplier.model_dump().items():
            setattr(db_supplier, key, value)
        db.commit()
        db.refresh(db_supplier)
        return db_supplier
    return None

def delete_supplier_db(db: Session, supplier_id: int) -> bool:
    db_supplier = db.query(Supplier).filter(Supplier.id == supplier_id).first()
    if db_supplier:
        db.delete(db_supplier)
        db.commit()
        return True
    return False

# CONSUMPTION CRUD OPERATIONS
def get_consumption_data(db: Session) -> List[Consumption]:
    return db.query(Consumption).all()

def create_consumption_record(db: Session, consumption: ConsumptionCreate) -> Consumption:
    # Note: consumption.project (not project_name) per our updated schema
    db_consumption = Consumption(
        material_id=consumption.material_id,
        project=consumption.project,
        quantity_used=consumption.quantity_used,
        date_used=consumption.date_used,
        notes=consumption.notes
    )
    db.add(db_consumption)
    db.commit()
    db.refresh(db_consumption)
    
    # Update inventory quantity based on consumption
    inventory = db.query(Inventory).filter(Inventory.id == consumption.material_id).first()
    if inventory:
        new_quantity = inventory.quantity - consumption.quantity_used
        update_inventory(db, inventory.id, new_quantity)
    
    return db_consumption

# COST CRUD OPERATIONS
def get_cost_data(db: Session) -> List[Cost]:
    return db.query(Cost).all()

def create_cost_record(db: Session, cost: CostCreate) -> Cost:
    cost_data = cost.model_dump()
    if not cost_data.get("total_cost"):
        cost_data["total_cost"] = cost_data["unit_price"] * cost_data["quantity_purchased"]
    if not cost_data.get("date_recorded"):
        cost_data["date_recorded"] = datetime.utcnow()
    db_cost = Cost(**cost_data)
    db.add(db_cost)
    db.commit()
    db.refresh(db_cost)
    return db_cost

# WASTE CRUD OPERATIONS
def get_waste_data(db: Session) -> List[Waste]:
    return db.query(Waste).all()

def create_waste_record(db: Session, waste: WasteCreate) -> Waste:
    db_waste = Waste(**waste.model_dump())
    db.add(db_waste)
    db.commit()
    db.refresh(db_waste)
    
    # Update inventory based on waste
    inventory = db.query(Inventory).filter(Inventory.id == waste.material_id).first()
    if inventory:
        update_inventory(db, inventory.id, inventory.quantity - waste.quantity_wasted)
    
    return db_waste

def get_waste_record(db: Session, waste_id: int) -> Optional[Waste]:
    return db.query(Waste).filter(Waste.id == waste_id).first()

# ALERT CRUD OPERATIONS
def get_alerts(db: Session) -> List[Alert]:
    return db.query(Alert).filter(Alert.is_active == 1).all()

def create_alert(db: Session, alert: AlertCreate) -> Alert:
    db_alert = Alert(**alert.model_dump())
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

def resolve_alert(db: Session, alert_id: int) -> Alert:
    db_alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if db_alert:
        db_alert.is_active = 0
        db.commit()
        db.refresh(db_alert)
    return db_alert
