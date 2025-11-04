# crud.py
import math
import statistics
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from models import Inventory, Supplier, Consumption, Cost, Waste, Alert, Project
from schemas import (
    InventoryCreate, SupplierCreate, SupplierUpdate,
    ConsumptionCreate, CostCreate, WasteCreate, AlertCreate, ProjectCreate, ProjectUpdate,
)
from broadcast import broadcast_manager
import asyncio

# ---------------------------
# Project CRUD operations
# ---------------------------
def get_project_by_id(db: Session, project_id: int) -> Optional[Project]:
    return db.query(Project).filter(Project.id == project_id).first()

def get_project_by_name(db: Session, name: str) -> Optional[Project]:
    return db.query(Project).filter(Project.name == name).first()

def get_projects(db: Session, skip: int = 0, limit: int = 100) -> List[Project]:
    return db.query(Project).order_by(Project.name).offset(skip).limit(limit).all()

def create_project(db: Session, project: ProjectCreate) -> Project:
    existing_project = get_project_by_name(db, project.name)
    if existing_project:
        raise IntegrityError(f"Project with name '{project.name}' already exists.", params=None, orig=None)
    db_project = Project(**project.model_dump())
    db.add(db_project)
    try:
        db.commit()
        db.refresh(db_project)
        return db_project
    except IntegrityError as e:
        db.rollback()
        print(f"IntegrityError creating project: {e}")
        raise IntegrityError(f"Database error creating project '{project.name}'.", params=None, orig=e)
    except Exception as e:
        db.rollback()
        print(f"Unexpected error creating project: {e}")
        raise

def update_project(db: Session, project_id: int, project_update: ProjectUpdate) -> Optional[Project]:
    db_project = get_project_by_id(db, project_id)
    if not db_project:
        return None
    update_data = project_update.model_dump(exclude_unset=True)
    if "name" in update_data and update_data["name"] != db_project.name:
        existing_project = get_project_by_name(db, update_data["name"])
        if existing_project and existing_project.id != project_id:
             raise IntegrityError(f"Another project with name '{update_data['name']}' already exists.", params=None, orig=None)
    for key, value in update_data.items():
        setattr(db_project, key, value)
    try:
        db.commit()
        db.refresh(db_project)
        return db_project
    except IntegrityError as e:
        db.rollback()
        print(f"IntegrityError updating project ID {project_id}: {e}")
        raise IntegrityError("Database error updating project.", params=None, orig=e)
    except Exception as e:
        db.rollback()
        print(f"Unexpected error updating project ID: {e}")
        raise

def delete_project(db: Session, project_id: int) -> bool:
    db_project = get_project_by_id(db, project_id)
    if db_project:
        if db_project.inventory_items:
            raise ValueError(f"Cannot delete project ID {project_id} as it has linked inventory items.")
        if db_project.consumption_records:
            raise ValueError(f"Cannot delete project ID {project_id} as it has linked consumption records.")
        if db_project.waste_records:
            raise ValueError(f"Cannot delete project ID {project_id} as it has linked waste records.")
        db.delete(db_project)
        db.commit()
        return True
    return False

# ---------------------------
# Supplier CRUD operations
# ---------------------------
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

def update_supplier_db(db: Session, supplier_id: int, supplier: SupplierUpdate) -> Optional[Supplier]:
    db_supplier = get_supplier_by_id(db, supplier_id)
    if db_supplier:
        update_data = supplier.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_supplier, key, value)
        db.commit()
        db.refresh(db_supplier)
        return db_supplier
    return None

def delete_supplier_db(db: Session, supplier_id: int) -> bool:
    db_supplier = get_supplier_by_id(db, supplier_id)
    if db_supplier:
        linked_materials = db.query(Inventory).filter(Inventory.supplier_id == supplier_id).first()
        if linked_materials:
             raise ValueError(f"Cannot delete supplier ID {supplier_id} as it is linked to materials.")
        db.delete(db_supplier)
        db.commit()
        return True
    return False

# ---------------------------
# Inventory CRUD operations
# ---------------------------
def get_inventory_item(db: Session, item_id: int) -> Optional[Inventory]:
    return db.query(Inventory).options(
        joinedload(Inventory.supplier),
        joinedload(Inventory.project)
    ).filter(Inventory.id == item_id).first()

def get_inventory(db: Session, project_id: Optional[int] = None) -> List[Inventory]:
    query = db.query(Inventory).options(
        joinedload(Inventory.supplier),
        joinedload(Inventory.project)
    )
    if project_id is not None:
        query = query.filter(Inventory.project_id == project_id)
    return query.order_by(Inventory.project_id, Inventory.material_name).all()

def create_inventory(db: Session, item: InventoryCreate) -> Inventory:
    project = get_project_by_id(db, item.project_id)
    if not project:
        raise ValueError(f"Project with ID {item.project_id} not found.")
    if item.supplier_id:
        supplier = get_supplier_by_id(db, item.supplier_id)
        if not supplier:
             raise ValueError(f"Supplier with ID {item.supplier_id} not found.")
    existing_item = db.query(Inventory).filter(
        Inventory.material_name == item.material_name,
        Inventory.project_id == item.project_id
    ).first()
    if existing_item:
        raise IntegrityError(f"Inventory item '{item.material_name}' already exists for Project ID {item.project_id}.", params=None, orig=None)

    # Compute estimated_total_value if not provided but qty and unit price are present
    item_data = item.model_dump()
    if item_data.get("estimated_total_value") is None:
        eq = item_data.get("estimated_quantity")
        eup = item_data.get("estimated_unit_price")
        if eq is not None and eup is not None:
            item_data["estimated_total_value"] = eq * eup

    db_item = Inventory(**item_data)
    db.add(db_item)
    try:
        db.commit()
        db.refresh(db_item)
        db.refresh(db_item, attribute_names=['project', 'supplier'])
        try:
            notify_payload = {
                "inventory_id": db_item.id,
                "material_name": db_item.material_name,
                "project_id": db_item.project_id
            }
            notify_broadcast("inventory_created", notify_payload)
        except Exception as e:
            print(f"Failed to notify broadcast for inventory creation: {e}")
        return db_item
    except IntegrityError as e:
        db.rollback()
        print(f"IntegrityError creating inventory item: {e}")
        raise IntegrityError(f"Database error creating inventory item '{item.material_name}'.", params=None, orig=e)
    except Exception as e:
        db.rollback()
        print(f"Unexpected error creating inventory item: {e}")
        raise

def update_inventory_quantity(db: Session, item_id: int, quantity: float) -> Optional[Inventory]:
    db_item = get_inventory_item(db, item_id)
    if db_item:
        db_item.quantity = quantity
        db_item.last_updated = datetime.utcnow()
        if quantity <= db_item.reorder_point:
            existing_alert = db.query(Alert).filter(
                Alert.material_id == item_id,
                Alert.alert_type == "low_stock",
                Alert.is_active
            ).first()
            if not existing_alert:
                project_name = db_item.project.name if db_item.project else "Unknown Project"
                create_alert(db, AlertCreate(
                    material_id=item_id,
                    alert_type="low_stock",
                    message=f"Material {db_item.material_name} (ID: {item_id}) in Project '{project_name}' is below reorder point ({db_item.reorder_point}). Current quantity: {quantity}."
                ))
        try:
            db.commit()
            db.refresh(db_item)
            db.refresh(db_item, attribute_names=['project', 'supplier'])
            try:
                notify_payload = {
                    "inventory_id": db_item.id,
                    "quantity": db_item.quantity,
                    "project_id": db_item.project_id
                }
                notify_broadcast("inventory_updated", notify_payload)
            except Exception as e:
                print(f"Failed to notify broadcast for inventory update: {e}")
            return db_item
        except Exception as e:
            db.rollback()
            print(f"Error committing inventory quantity update for item {item_id}: {e}")
            raise
    return None

def update_inventory_item_details(db: Session, item_id: int, item_update: InventoryCreate) -> Optional[Inventory]:
    db_item = get_inventory_item(db, item_id)
    if not db_item:
        return None
    update_data = item_update.model_dump(exclude_unset=True, exclude={'quantity'})
    if "project_id" in update_data and update_data["project_id"] != db_item.project_id:
        if not get_project_by_id(db, update_data["project_id"]):
            raise ValueError(f"Project with ID {update_data['project_id']} not found.")
    if "supplier_id" in update_data and update_data["supplier_id"] != db_item.supplier_id:
         if update_data["supplier_id"] is not None and not get_supplier_by_id(db, update_data["supplier_id"]):
             raise ValueError(f"Supplier with ID {update_data['supplier_id']} not found.")

    # Compute estimated_total_value if not provided but other fields available
    if update_data.get("estimated_total_value") is None:
        eq = update_data.get("estimated_quantity", db_item.estimated_quantity)
        eup = update_data.get("estimated_unit_price", db_item.estimated_unit_price)
        if eq is not None and eup is not None:
            update_data["estimated_total_value"] = eq * eup

    new_name = update_data.get("material_name", db_item.material_name)
    new_project_id = update_data.get("project_id", db_item.project_id)
    if new_name != db_item.material_name or new_project_id != db_item.project_id:
        existing_item = db.query(Inventory).filter(
            Inventory.material_name == new_name,
            Inventory.project_id == new_project_id,
            Inventory.id != item_id
        ).first()
        if existing_item:
            raise IntegrityError(f"Inventory item '{new_name}' already exists for Project ID {new_project_id}.", params=None, orig=None)

    for key, value in update_data.items():
        setattr(db_item, key, value)
    db_item.last_updated = datetime.utcnow()
    try:
        db.commit()
        db.refresh(db_item)
        db.refresh(db_item, attribute_names=['project', 'supplier'])
        return db_item
    except IntegrityError as e:
        db.rollback()
        print(f"IntegrityError updating inventory item details {item_id}: {e}")
        raise IntegrityError(f"Database error updating inventory item '{new_name}'.", params=None, orig=e)
    except Exception as e:
        db.rollback()
        print(f"Unexpected error updating inventory item details {item_id}: {e}")
        raise

def delete_inventory_item(db: Session, item_id: int) -> bool:
    db_item = get_inventory_item(db, item_id)
    if db_item:
        db.delete(db_item)
        db.commit()
        return True
    return False

# ---------------------------
# Consumption CRUD operations
# ---------------------------
def get_consumption_data(db: Session, project_id: Optional[int] = None) -> List[Consumption]:
    query = db.query(Consumption).options(
        joinedload(Consumption.material),
        joinedload(Consumption.project_rel)
    )
    if project_id is not None:
        query = query.filter(Consumption.project_id == project_id)
    return query.order_by(Consumption.date_used.desc()).all()

def create_consumption_record(db: Session, consumption: ConsumptionCreate) -> Consumption:
    inventory_item = get_inventory_item(db, consumption.material_id)
    if not inventory_item:
        raise ValueError(f"Material with ID {consumption.material_id} not found in inventory.")
    # Determine project_id (infer if omitted)
    if consumption.project_id is None:
        project_id_to_use = inventory_item.project_id
    else:
        project_id_to_use = consumption.project_id
        if inventory_item.project_id != project_id_to_use:
            raise ValueError(
                f"Material ID {consumption.material_id} belongs to Project ID {inventory_item.project_id}. "
                f"Provided project_id {project_id_to_use} does not match. Either omit project_id to use the material's project or provide the correct project ID."
            )
    project = get_project_by_id(db, project_id_to_use)
    if not project:
        raise ValueError(f"Project with ID {project_id_to_use} not found.")
    if inventory_item.quantity < consumption.quantity_used:
         raise ValueError(
             f"Insufficient inventory for material '{inventory_item.material_name}' in Project '{project.name}'. "
             f"Available: {inventory_item.quantity}, Needed: {consumption.quantity_used}"
         )
    db_consumption = Consumption(
        material_id=consumption.material_id,
        project_id=project_id_to_use,
        quantity_used=consumption.quantity_used,
        date_used=consumption.date_used or datetime.utcnow(),
        notes=consumption.notes
    )
    db.add(db_consumption)
    new_quantity = inventory_item.quantity - consumption.quantity_used
    inventory_item.quantity = new_quantity
    inventory_item.last_updated = datetime.utcnow()
    try:
        db.commit()
        # low-stock alert
        if new_quantity <= inventory_item.reorder_point:
             existing_alert = db.query(Alert).filter(Alert.material_id == inventory_item.id, Alert.alert_type == "low_stock", Alert.is_active).first()
             if not existing_alert:
                 create_alert(db, AlertCreate( material_id=inventory_item.id, alert_type="low_stock", message=f"Material {inventory_item.material_name} (ID: {inventory_item.id}) in Project '{project.name}' is below reorder point ({inventory_item.reorder_point}). Current quantity: {new_quantity}." ))
        # check limit exceeded alert
        db.refresh(inventory_item)
        check_and_create_limit_alert(db, inventory_item)
        db.refresh(db_consumption)
        db.refresh(db_consumption, attribute_names=['material', 'project_rel'])
        try:
            # notify frontends that a new consumption record exists
            notify_payload = {
                "consumption_id": db_consumption.id,
                "material_id": db_consumption.material_id,
                "material_name": db_consumption.material.material_name if db_consumption.material else None,
                "quantity_used": db_consumption.quantity_used,
                "date_used": db_consumption.date_used.isoformat(),
                "project_id": db_consumption.project_id
            }
            notify_broadcast("consumption_created", notify_payload)
        except Exception as e:
            print(f"Failed to notify broadcast for consumption: {e}")
        return db_consumption
    except Exception as e:
        db.rollback()
        print(f"Error during consumption creation/inventory update commit: {e}")
        raise

# ---------------------------
# Waste CRUD operations
# ---------------------------
def get_waste_data(db: Session, project_id: Optional[int] = None) -> List[Waste]:
    query = db.query(Waste).options(
        joinedload(Waste.material),
        joinedload(Waste.project_rel)
    )
    if project_id is not None:
        query = query.filter(Waste.project_id == project_id)
    return query.order_by(Waste.date_recorded.desc()).all()

def create_waste_record(db: Session, waste: WasteCreate) -> Waste:
    project = get_project_by_id(db, waste.project_id)
    if not project:
         raise ValueError(f"Project with ID {waste.project_id} not found.")
    inventory_item = db.query(Inventory).filter(
        Inventory.id == waste.material_id,
        Inventory.project_id == waste.project_id
    ).first()
    if not inventory_item:
        raise ValueError(f"Material with ID {waste.material_id} not found in Project ID {waste.project_id}.")
    db_waste = Waste(
        material_id=waste.material_id,
        project_id=waste.project_id,
        quantity_wasted=waste.quantity_wasted,
        reason=waste.reason,
        preventive_measures=waste.preventive_measures,
        date_recorded=datetime.utcnow()
    )
    db.add(db_waste)
    new_quantity = inventory_item.quantity - waste.quantity_wasted
    if new_quantity < 0:
        print(f"Warning: Logging waste for material ID {waste.material_id} in project '{project.name}' resulted in negative stock. Setting stock to 0.")
        new_quantity = 0
    inventory_item.quantity = new_quantity
    inventory_item.last_updated = datetime.utcnow()
    try:
        db.commit()
        if new_quantity <= inventory_item.reorder_point:
             existing_alert = db.query(Alert).filter(Alert.material_id == inventory_item.id, Alert.alert_type == "low_stock", Alert.is_active).first()
             if not existing_alert:
                 create_alert(db, AlertCreate( material_id=inventory_item.id, alert_type="low_stock", message=f"Material {inventory_item.material_name} (ID: {inventory_item.id}) in Project '{project.name}' is below reorder point ({inventory_item.reorder_point}). Current quantity: {new_quantity}." ))
        # check limit exceeded alert
        db.refresh(inventory_item)
        check_and_create_limit_alert(db, inventory_item)
        db.refresh(db_waste)
        db.refresh(db_waste, attribute_names=['material', 'project_rel'])
        try:
            notify_payload = {
                "waste_id": db_waste.id,
                "material_id": db_waste.material_id,
                "quantity_wasted": db_waste.quantity_wasted,
                "project_id": db_waste.project_id
            }
            notify_broadcast("waste_created", notify_payload)
        except Exception as e:
            print(f"Failed to notify broadcast for waste: {e}")
        return db_waste
    except Exception as e:
        db.rollback()
        print(f"Error during waste creation/inventory update commit: {e}")
        raise

def get_waste_record(db: Session, waste_id: int) -> Optional[Waste]:
    return db.query(Waste).options(
        joinedload(Waste.material),
        joinedload(Waste.project_rel)
    ).filter(Waste.id == waste_id).first()

# ---------------------------
# Cost CRUD operations
# ---------------------------
def get_cost_data(db: Session) -> List[Cost]:
    return db.query(Cost).options(
        joinedload(Cost.material),
        joinedload(Cost.supplier)
    ).order_by(Cost.date_recorded.desc()).all()

def create_cost_record(db: Session, cost: CostCreate) -> Cost:
    cost_data = cost.model_dump()
    inventory_item = get_inventory_item(db, cost.material_id)
    if not inventory_item:
         raise ValueError(f"Material with ID {cost.material_id} not found in inventory.")
    if cost.supplier_id and not get_supplier_by_id(db, cost.supplier_id):
         raise ValueError(f"Supplier with ID {cost.supplier_id} not found.")
    if cost_data.get("total_cost") is None:
        unit_price = cost_data.get("unit_price")
        quantity_purchased = cost_data.get("quantity_purchased")
        if unit_price is not None and quantity_purchased is not None:
             cost_data["total_cost"] = unit_price * quantity_purchased
        else:
             missing_fields = []
             if unit_price is None:
                 missing_fields.append("unit_price")
             if quantity_purchased is None:
                 missing_fields.append("quantity_purchased")
             raise ValueError(f"Cannot calculate total_cost: Required fields missing: {', '.join(missing_fields)}")
    elif cost_data["total_cost"] < 0:
         raise ValueError("Total cost cannot be negative.")
    cost_data.setdefault("date_recorded", datetime.utcnow())
    if isinstance(cost_data.get("date_recorded"), str):
        try:
            from dateutil import parser
            cost_data["date_recorded"] = parser.parse(cost_data["date_recorded"])
        except (ImportError, ValueError):
             print(f"Warning: Could not parse date string '{cost_data.get('date_recorded')}'. Using current UTC time.")
             cost_data["date_recorded"] = datetime.utcnow()
    db_cost = Cost(**cost_data)
    db.add(db_cost)
    try:
        db.commit()
        db.refresh(db_cost)
        db.refresh(db_cost, attribute_names=['material', 'supplier'])
        # check limit exceeded for the related inventory item
        db.refresh(inventory_item)
        check_and_create_limit_alert(db, inventory_item)
        try:
            notify_payload = {
                "cost_id": db_cost.id,
                "material_id": db_cost.material_id,
                "total_cost": db_cost.total_cost,
                "project_id": inventory_item.project_id
            }
            notify_broadcast("cost_created", notify_payload)
        except Exception as e:
            print(f"Failed to notify broadcast for cost: {e}")
        return db_cost
    except IntegrityError as e:
        db.rollback()
        if hasattr(e, 'orig') and "NOT NULL constraint failed: costs.total_cost" in str(e.orig):
             print(f"IntegrityError: total_cost was unexpectedly NULL before commit. Data: {cost_data}")
        else:
             print(f"IntegrityError creating cost record: {e}")
        raise
    except Exception as e:
        db.rollback()
        print(f"Unexpected error creating cost record: {e}")
        raise

# ---------------------------
# Alerts
# ---------------------------
def get_alerts(db: Session) -> List[Alert]:
    return db.query(Alert).options(
        joinedload(Alert.material).joinedload(Inventory.project)
        ).filter(Alert.is_active).order_by(Alert.date_created.desc()).all()

def create_alert(db: Session, alert: AlertCreate) -> Alert:
    existing_alert = db.query(Alert).filter(
        Alert.material_id == alert.material_id,
        Alert.alert_type == alert.alert_type,
        Alert.is_active
    ).first()
    if existing_alert:
        print(f"Skipping duplicate active alert creation for material ID {alert.material_id}, type {alert.alert_type}")
        return existing_alert
    db_alert = Alert(**alert.model_dump())
    db.add(db_alert)
    try:
        db.commit()
        db.refresh(db_alert)
        try:
            notify_payload = {
                "alert_id": db_alert.id,
                "material_id": db_alert.material_id,
                "alert_type": db_alert.alert_type,
                "message": db_alert.message
            }
            notify_broadcast("alert_created", notify_payload)
        except Exception as e:
            print(f"Failed to notify broadcast for alert: {e}")
        return db_alert
    except Exception as e:
         db.rollback()
         print(f"Error creating alert: {e}")
         return None

def resolve_alert(db: Session, alert_id: int) -> Optional[Alert]:
    db_alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if db_alert:
        db_alert.is_active = False
        db.commit()
        db.refresh(db_alert)
        return db_alert
    return None

# ---------------------------
# Resource-limit helpers & reports
# ---------------------------
def compute_estimated_total_value(inventory_item: Inventory) -> Optional[float]:
    """
    Returns the estimated total value if available:
      - if estimated_total_value column is set, return it
      - else if estimated_quantity & estimated_unit_price present, compute product
      - else return None
    """
    if inventory_item.estimated_total_value is not None:
        try:
            return float(inventory_item.estimated_total_value)
        except Exception:
            return None
    if inventory_item.estimated_quantity is not None and inventory_item.estimated_unit_price is not None:
        try:
            return float(inventory_item.estimated_quantity) * float(inventory_item.estimated_unit_price)
        except Exception:
            return None
    return None

def compute_actual_consumed_quantity(db: Session, inventory_id: int) -> float:
    cons_sum = db.query(func.coalesce(func.sum(Consumption.quantity_used), 0.0)).filter(Consumption.material_id == inventory_id).scalar() or 0.0
    waste_sum = db.query(func.coalesce(func.sum(Waste.quantity_wasted), 0.0)).filter(Waste.material_id == inventory_id).scalar() or 0.0
    try:
        return float(cons_sum) + float(waste_sum)
    except Exception:
        return 0.0

def compute_actual_value_used(db: Session, inventory_item: Inventory) -> float:
    # Prefer accumulated cost totals if present
    cost_sum = db.query(func.coalesce(func.sum(Cost.total_cost), 0.0)).filter(Cost.material_id == inventory_item.id).scalar() or 0.0
    try:
        cost_sum = float(cost_sum)
    except Exception:
        cost_sum = 0.0
    if cost_sum > 0:
        return cost_sum
    # Fallback: use estimated_unit_price (or latest cost unit_price) * consumed quantity
    unit_price = inventory_item.estimated_unit_price
    if unit_price is None:
        latest_cost = db.query(Cost).filter(Cost.material_id == inventory_item.id).order_by(Cost.date_recorded.desc()).first()
        if latest_cost:
            unit_price = latest_cost.unit_price
    if unit_price is None:
        return 0.0
    consumed_qty = compute_actual_consumed_quantity(db, inventory_item.id)
    try:
        return float(unit_price) * float(consumed_qty)
    except Exception:
        return 0.0

def check_and_create_limit_alert(db: Session, inventory_item: Inventory):
    """
    Compare actual value used with estimate and create/deactivate limit_exceeded alerts.
    """
    estimated = compute_estimated_total_value(inventory_item)
    if estimated is None:
        # no estimate; nothing to check
        return None
    actual_value = compute_actual_value_used(db, inventory_item)
    if actual_value > estimated:
        project_name = inventory_item.project.name if inventory_item.project else "Unknown Project"
        message = (f"Material {inventory_item.material_name} (ID: {inventory_item.id}) in Project '{project_name}' "
                   f"exceeded estimated value. Estimated: {estimated:.2f}, Actual: {actual_value:.2f} (over by {actual_value - estimated:.2f}).")
        existing = db.query(Alert).filter(
            Alert.material_id == inventory_item.id,
            Alert.alert_type == "limit_exceeded",
            Alert.is_active
        ).first()
        if existing:
            # Already active — optionally update message/time — skip update to avoid extra commits
            return existing
        try:
            # create a new alert record using same Alert model
            db_alert = Alert(material_id=inventory_item.id, alert_type="limit_exceeded", message=message)
            db.add(db_alert)
            db.commit()
            db.refresh(db_alert)
            return db_alert
        except Exception as e:
            db.rollback()
            print(f"Failed creating limit_exceeded alert: {e}")
            return None
    else:
        # If previously there was an active limit_exceeded alert, deactivate it (resolved)
        existing = db.query(Alert).filter(Alert.material_id == inventory_item.id, Alert.alert_type == "limit_exceeded", Alert.is_active).first()
        if existing:
            try:
                existing.is_active = False
                db.commit()
                db.refresh(existing)
            except Exception as e:
                db.rollback()
                print(f"Failed to resolve old limit_exceeded alert: {e}")
        return None

def get_resource_limits_report(db: Session, project_id: int):
    """
    Returns a dict: { items: [...], summary: {...} } for the given project.
    Each item includes estimated_total_value, actual_value_used, variance, percent and status.
    """
    items = get_inventory(db, project_id=project_id)
    report_items = []
    for item in items:
        est = compute_estimated_total_value(item)
        actual = compute_actual_value_used(db, item)
        if est is None:
            status = "no_estimate"
            variance = None
            pct = None
        else:
            variance = float(actual) - float(est)
            pct = (variance / est) * 100 if est != 0 else None
            status = "exceeded" if actual > est else "under"
        report_items.append({
            "material_id": item.id,
            "material_name": item.material_name,
            "project_id": item.project_id,
            "estimated_total_value": None if est is None else round(est, 2),
            "actual_value_used": round(actual, 2),
            "variance_value": None if variance is None else round(variance, 2),
            "variance_percent": None if pct is None else round(pct, 2),
            "status": status,
            "estimated_quantity": item.estimated_quantity,
            "estimated_unit_price": item.estimated_unit_price
        })
    summary = {
        "total_items": len(report_items),
        "num_exceeded": sum(1 for r in report_items if r["status"] == "exceeded"),
        "num_under": sum(1 for r in report_items if r["status"] == "under"),
        "num_no_estimate": sum(1 for r in report_items if r["status"] == "no_estimate")
    }
    # Optionally sort report_items by variance_percent desc for easy consumption
    report_items_sorted = sorted(report_items, key=lambda r: (r["variance_percent"] if r["variance_percent"] is not None else -9999), reverse=True)
    return {"items": report_items_sorted, "summary": summary}

# ---------- Procurement advisor helpers ----------

def _get_consumption_history(db: Session, material_id: int, days: int = 90):
    """
    Return list of daily consumption quantities for the last `days` days.
    We'll query Consumption.date_used and sum quantity_used by day.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    rows = db.query(
        func.date(Consumption.date_used).label('d'),
        func.coalesce(func.sum(Consumption.quantity_used), 0).label('qty')
    ).filter(
        Consumption.material_id == material_id,
        Consumption.date_used >= cutoff
    ).group_by(func.date(Consumption.date_used)).order_by(func.date(Consumption.date_used)).all()
    # rows -> list of (d, qty)
    return [float(r.qty) for r in rows]

def estimate_daily_demand(consumption_series: list):
    """Return average daily demand and std dev from series (list of daily quantities)."""
    if not consumption_series:
        return 0.0, 0.0
    avg = statistics.mean(consumption_series)
    stdev = statistics.pstdev(consumption_series) if len(consumption_series) > 1 else 0.0
    return float(avg), float(stdev)

def compute_eoq(D, S, H):
    """
    EOQ = sqrt(2 * D * S / H)
    D = demand rate (units per year) - we'll annualize daily demand * 365
    S = ordering/setup cost per order (we'll use a default or config)
    H = holding cost per unit per year (use % of unit price or config)
    Returns EOQ (float) or None if insufficient data.
    """
    try:
        if D <= 0 or S <= 0 or H <= 0:
            return None
        return math.sqrt((2.0 * D * S) / H)
    except Exception:
        return None

def compute_safety_stock(avg_daily_demand, demand_std_dev, lead_time_days, lead_time_std=0.0, service_level_z=1.65):
    """
    Basic safety stock formula (covering demand variability):
      safety_stock = Z * sigma_demand * sqrt(lead_time)
    More advanced: include lead time variability: Z * sqrt((sigma_demand^2 * L) + (avg_demand^2 * sigma_lead^2))
    We'll implement the latter if lead_time_std provided.
    service_level_z: default about 95% -> ~1.65
    """
    try:
        L = max(lead_time_days, 1e-6)
        sigma_d = float(demand_std_dev)
        sigma_L = float(lead_time_std or 0.0)
        mu = float(avg_daily_demand)
        if sigma_L and sigma_L > 0:
            safety = service_level_z * math.sqrt((sigma_d ** 2) * L + (mu ** 2) * (sigma_L ** 2))
        else:
            safety = service_level_z * sigma_d * math.sqrt(L)
        return float(safety)
    except Exception:
        return 0.0

def compute_reorder_point(avg_daily_demand, lead_time_days, safety_stock):
    # Demand during lead time + safety stock
    return float(avg_daily_demand * lead_time_days + (safety_stock or 0.0))

def rank_suppliers_for_material(db: Session, material_id: int):
    """
    Return list of suppliers with a composite score based on:
      - avg unit price (lower better)
      - lead_time_days (lower better)
      - reliability_rating (higher better)
    If supplier has no price, we still include them but deprioritize.
    """
    # gather suppliers linked to this material (inventory entries across projects)
    suppliers = db.query(Supplier).all()
    scored = []
    for s in suppliers:
        # estimate avg unit price for this supplier for this material (from Cost records)
        avg_price_row = db.query(func.coalesce(func.avg(Cost.unit_price), None)).filter(Cost.supplier_id == s.id, Cost.material_id == material_id).scalar()
        avg_price = float(avg_price_row) if avg_price_row else None
        lead = s.lead_time_days
        rel = s.reliability_rating if s.reliability_rating is not None else 0.0
        # compute score: higher is better
        # Normalize components with simple heuristics
        price_score = (1.0 / avg_price) if avg_price and avg_price > 0 else 0.0
        lead_score = (1.0 / (lead + 1)) if lead is not None else 0.0
        rel_score = rel / 5.0  # 0..1
        # weights: reliability 40%, price 35%, lead 25%
        score = 0.4 * rel_score + 0.35 * price_score + 0.25 * lead_score
        scored.append({
            "supplier_id": s.id,
            "supplier_name": s.name,
            "lead_time_days": lead,
            "reliability_rating": rel,
            "avg_unit_price": avg_price,
            "score": round(score, 4)
        })
    # sort desc
    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored_sorted

def get_procurement_suggestions(db: Session, project_id: int, lookback_days: int = 90, service_level_z: float = 1.65):
    """
    Returns list of suggestion dicts for items in the project:
      - current stock, daily demand, stddev, safety_stock, ROP, recommended order qty (EOQ or ROP*some factor)
    """
    project = get_project_by_id(db, project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")
    items = get_inventory(db, project_id=project_id)
    suggestions = []
    for it in items:
        # get consumption history
        series = _get_consumption_history(db, it.id, days=lookback_days)
        avg_daily, stdev_daily = estimate_daily_demand(series)
        # lead time default: supplier lead_time if supplier exists else 7 days
        lead_time = it.supplier.lead_time_days if it.supplier and it.supplier.lead_time_days else 7
        # lead time std: assume 1/3 of lead_time as rough proxy if not provided
        lead_time_std = max(1.0, lead_time * 0.33)
        safety = compute_safety_stock(avg_daily, stdev_daily, lead_time, lead_time_std, service_level_z=service_level_z)
        reorder_pt = compute_reorder_point(avg_daily, lead_time, safety)
        days_of_stock = (it.quantity / avg_daily) if avg_daily > 0 else None
        # EOQ inputs: annual demand D, ordering cost S (default), holding cost H (default using estimated unit price)
        D = avg_daily * 365.0
        S = 200.0  # default order cost in currency units — you may make this configurable
        # holding cost: assume 20% of unit price per year if unit price known
        unit_price = None
        # prefer latest cost unit_price
        latest_cost = db.query(Cost).filter(Cost.material_id == it.id).order_by(Cost.date_recorded.desc()).first()
        if latest_cost:
            unit_price = latest_cost.unit_price
        elif it.estimated_unit_price:
            unit_price = it.estimated_unit_price
        H = (0.2 * unit_price) if unit_price is not None else None
        eoq = compute_eoq(D, S, H) if H else None
        # fallback recommended qty: max( eoq, reorder_pt * 1.5 ) or at least to cover lead time + safety
        rec_qty = None
        if eoq and eoq > 0:
            rec_qty = max(eoq, reorder_pt * 1.5)
        else:
            # fallback order enough to refill to estimated quantity or to cover lead time*avg + safety
            target = it.estimated_quantity if it.estimated_quantity else max(avg_daily * lead_time * 3, reorder_pt * 2)
            rec_qty = max(0.0, target - it.quantity)
        # supplier ranking
        suppliers_ranked = rank_suppliers_for_material(db, it.id)
        # short reason
        reason = f"Avg daily demand {avg_daily:.2f}, stdev {stdev_daily:.2f}, lead_time {lead_time}d. Reorder point {reorder_pt:.2f}."
        suggestions.append({
            "material_id": it.id,
            "material_name": it.material_name,
            "current_quantity": float(it.quantity),
            "days_of_stock": None if days_of_stock is None else round(days_of_stock, 2),
            "recommended_order_qty": round(float(rec_qty), 2) if rec_qty is not None else None,
            "reorder_point": round(float(reorder_pt), 2),
            "safety_stock": round(float(safety), 2),
            "reason": reason,
            "supplier_scores": suppliers_ranked[:5]  # top 5
        })
    return {
        "project_id": project.id,
        "project_name": project.name,
        "generated_at": datetime.utcnow(),
        "suggestions": suggestions
    }

# --- Dashboard aggregator helpers (used by snapshot endpoint) ---
def crud_get_total_materials(db: Session, project_id: Optional[int] = None) -> int:
    q = db.query(func.count(Inventory.id))
    if project_id:
        q = q.filter(Inventory.project_id == project_id)
    return int(q.scalar() or 0)

def crud_get_active_alerts_count(db: Session, project_id: Optional[int] = None) -> int:
    q = db.query(func.count(Alert.id)).filter(Alert.is_active)
    if project_id:
        # alerts link to material -> inventory has project_id
        q = q.join(Inventory, Alert.material_id == Inventory.id).filter(Inventory.project_id == project_id)
    return int(q.scalar() or 0)

def crud_get_num_exceeded(db: Session, project_id: Optional[int] = None) -> int:
    # count of inventory items where actual > estimate
    items = get_inventory(db, project_id=project_id)
    cnt = 0
    for it in items:
        est = compute_estimated_total_value(it)
        if est is None:
            continue
        actual = compute_actual_value_used(db, it)
        if actual > est:
            cnt += 1
    return cnt

def crud_get_total_cost(db: Session, project_id: Optional[int] = None) -> float:
    q = db.query(func.coalesce(func.sum(Cost.total_cost), 0.0)).join(Inventory, Cost.material_id == Inventory.id)
    if project_id:
        q = q.filter(Inventory.project_id == project_id)
    val = q.scalar() or 0.0
    return float(val)


def crud_get_recent_consumption(db: Session, project_id: Optional[int] = None, limit: int = 10):
    q = db.query(Consumption).options(joinedload(Consumption.material)).order_by(Consumption.date_used.desc())
    if project_id:
        q = q.filter(Consumption.project_id == project_id)
    rows = q.limit(limit).all()
    # serialize minimal fields
    return [{
        "id": r.id,
        "material_id": r.material_id,
        "material_name": r.material.material_name if r.material else None,
        "quantity_used": r.quantity_used,
        "date_used": r.date_used.isoformat(),
        "project_id": r.project_id
    } for r in rows]


def crud_get_top_overruns(db: Session, project_id: Optional[int] = None, limit: int = 5):
    report = get_resource_limits_report(db, project_id) if project_id is not None else None
    if report and "items" in report:
        # filter only exceeded
        exceeded = [i for i in report["items"] if i["status"] == "exceeded"]
        return exceeded[:limit]
    # fallback: compute across all inventory if no project passed
    items = get_inventory(db, project_id=project_id)
    overruns = []
    for it in items:
        est = compute_estimated_total_value(it)
        if est is None:
            continue
        actual = compute_actual_value_used(db, it)
        if actual > est:
            overruns.append({
                "material_id": it.id,
                "material_name": it.material_name,
                "estimated_total_value": round(est,2),
                "actual_value_used": round(actual,2),
                "variance": round(actual - est,2)
            })
    overruns_sorted = sorted(overruns, key=lambda x: x["variance"], reverse=True)
    return overruns_sorted[:limit]

# --- Broadcast notifier used by CRUD functions after DB commit ---
def notify_broadcast(event_type: str, payload: dict):
    """
    Fire-and-forget broadcast via global broadcast_manager.
    event_type: strings like "consumption_created", "cost_created", "inventory_updated", "alert_created"
    payload: small JSON serializable dict
    """
    # schedule background broadcast on event loop
    try:
        coro = broadcast_manager.broadcast_json({"type": event_type, "payload": payload})
        # use asyncio.create_task if within async context; otherwise run in new loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(coro)
        else:
            # for synchronous contexts (rare), run until complete
            loop.run_until_complete(coro)
    except Exception as e:
        print("notify_broadcast failed:", e)
