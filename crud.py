# # crud.py

# from sqlalchemy.orm import Session, joinedload
# from sqlalchemy.exc import IntegrityError
# from datetime import datetime
# from typing import List, Optional

# from models import Inventory, Supplier, Consumption, Cost, Waste, Alert, Project
# from schemas import (
#     InventoryCreate, InventoryRead,
#     SupplierCreate, SupplierRead, SupplierUpdate,
#     ConsumptionCreate, ConsumptionRead,
#     CostCreate, CostRead,
#     WasteCreate, WasteRead,
#     AlertCreate, AlertRead,
#     ProjectCreate, ProjectUpdate, ProjectRead
# )

# # --- Project CRUD Operations ---
# def get_project_by_id(db: Session, project_id: int) -> Optional[Project]:
#     return db.query(Project).filter(Project.id == project_id).first()

# def get_project_by_name(db: Session, name: str) -> Optional[Project]:
#     return db.query(Project).filter(Project.name == name).first()

# def get_projects(db: Session, skip: int = 0, limit: int = 100) -> List[Project]:
#     return db.query(Project).order_by(Project.name).offset(skip).limit(limit).all()

# def create_project(db: Session, project: ProjectCreate) -> Project:
#     existing_project = get_project_by_name(db, project.name)
#     if existing_project:
#         raise IntegrityError(f"Project with name '{project.name}' already exists.", params=None, orig=None)

#     db_project = Project(**project.model_dump())
#     db.add(db_project)
#     try:
#         db.commit()
#         db.refresh(db_project)
#         return db_project
#     except IntegrityError as e:
#         db.rollback()
#         print(f"IntegrityError creating project: {e}")
#         raise IntegrityError(f"Database error creating project '{project.name}'.", params=None, orig=e)
#     except Exception as e:
#         db.rollback()
#         print(f"Unexpected error creating project: {e}")
#         raise

# def update_project(db: Session, project_id: int, project_update: ProjectUpdate) -> Optional[Project]:
#     db_project = get_project_by_id(db, project_id)
#     if not db_project:
#         return None

#     update_data = project_update.model_dump(exclude_unset=True)

#     if "name" in update_data and update_data["name"] != db_project.name:
#         existing_project = get_project_by_name(db, update_data["name"])
#         if existing_project and existing_project.id != project_id:
#              raise IntegrityError(f"Another project with name '{update_data['name']}' already exists.", params=None, orig=None)

#     for key, value in update_data.items():
#         setattr(db_project, key, value)

#     try:
#         db.commit()
#         db.refresh(db_project)
#         return db_project
#     except IntegrityError as e:
#         db.rollback()
#         print(f"IntegrityError updating project ID {project_id}: {e}")
#         raise IntegrityError(f"Database error updating project '{update_data.get('name', db_project.name)}'.", params=None, orig=e)
#     except Exception as e:
#         db.rollback()
#         print(f"Unexpected error updating project ID: {e}")
#         raise

# def delete_project(db: Session, project_id: int) -> bool:
#     db_project = get_project_by_id(db, project_id)
#     if db_project:
#         if db_project.inventory_items:
#             raise ValueError(f"Cannot delete project ID {project_id} as it has linked inventory items.")
#         if db_project.consumption_records:
#             raise ValueError(f"Cannot delete project ID {project_id} as it has linked consumption records.")
#         if db_project.waste_records:
#             raise ValueError(f"Cannot delete project ID {project_id} as it has linked waste records.")
#         db.delete(db_project)
#         db.commit()
#         return True
#     return False

# # --- Supplier CRUD Operations ---
# def get_suppliers(db: Session) -> List[Supplier]:
#     return db.query(Supplier).all()

# def create_supplier_db(db: Session, supplier: SupplierCreate) -> Supplier:
#     db_supplier = Supplier(**supplier.model_dump())
#     db.add(db_supplier)
#     db.commit()
#     db.refresh(db_supplier)
#     return db_supplier

# def get_supplier_by_id(db: Session, supplier_id: int) -> Optional[Supplier]:
#     return db.query(Supplier).filter(Supplier.id == supplier_id).first()

# def update_supplier_db(db: Session, supplier_id: int, supplier: SupplierUpdate) -> Optional[Supplier]:
#     db_supplier = get_supplier_by_id(db, supplier_id)
#     if db_supplier:
#         update_data = supplier.model_dump(exclude_unset=True)
#         for key, value in update_data.items():
#             setattr(db_supplier, key, value)
#         db.commit()
#         db.refresh(db_supplier)
#         return db_supplier
#     return None

# def delete_supplier_db(db: Session, supplier_id: int) -> bool:
#     db_supplier = get_supplier_by_id(db, supplier_id)
#     if db_supplier:
#         linked_materials = db.query(Inventory).filter(Inventory.supplier_id == supplier_id).first()
#         if linked_materials:
#              raise ValueError(f"Cannot delete supplier ID {supplier_id} as it is linked to materials.")
#         db.delete(db_supplier)
#         db.commit()
#         return True
#     return False

# # --- Inventory CRUD Operations (Updated) ---
# def get_inventory_item(db: Session, item_id: int) -> Optional[Inventory]:
#     return db.query(Inventory).options(
#         joinedload(Inventory.supplier),
#         joinedload(Inventory.project)
#     ).filter(Inventory.id == item_id).first()

# def get_inventory(db: Session, project_id: Optional[int] = None) -> List[Inventory]:
#     query = db.query(Inventory).options(
#         joinedload(Inventory.supplier),
#         joinedload(Inventory.project)
#     )
#     if project_id is not None:
#         query = query.filter(Inventory.project_id == project_id)
#     return query.order_by(Inventory.project_id, Inventory.material_name).all()

# def create_inventory(db: Session, item: InventoryCreate) -> Inventory:
#     project = get_project_by_id(db, item.project_id)
#     if not project:
#         raise ValueError(f"Project with ID {item.project_id} not found.")
#     if item.supplier_id:
#         supplier = get_supplier_by_id(db, item.supplier_id)
#         if not supplier:
#              raise ValueError(f"Supplier with ID {item.supplier_id} not found.")

#     existing_item = db.query(Inventory).filter(
#         Inventory.material_name == item.material_name,
#         Inventory.project_id == item.project_id
#     ).first()
#     if existing_item:
#         raise IntegrityError(f"Inventory item '{item.material_name}' already exists for Project ID {item.project_id}.", params=None, orig=None)

#     db_item = Inventory(**item.model_dump())
#     db.add(db_item)
#     try:
#         db.commit()
#         db.refresh(db_item)
#         db.refresh(db_item, attribute_names=['project', 'supplier'])
#         return db_item
#     except IntegrityError as e:
#         db.rollback()
#         print(f"IntegrityError creating inventory item: {e}")
#         raise IntegrityError(f"Database error creating inventory item '{item.material_name}'.", params=None, orig=e)
#     except Exception as e:
#         db.rollback()
#         print(f"Unexpected error creating inventory item: {e}")
#         raise

# def update_inventory_quantity(db: Session, item_id: int, quantity: float) -> Optional[Inventory]:
#     db_item = get_inventory_item(db, item_id)
#     if db_item:
#         db_item.quantity = quantity
#         db_item.last_updated = datetime.utcnow()
#         if quantity <= db_item.reorder_point:
#             existing_alert = db.query(Alert).filter(
#                 Alert.material_id == item_id,
#                 Alert.alert_type == "low_stock",
#                 Alert.is_active == True
#             ).first()
#             if not existing_alert:
#                 project_name = db_item.project.name if db_item.project else "Unknown Project"
#                 create_alert(db, AlertCreate(
#                     material_id=item_id,
#                     alert_type="low_stock",
#                     message=f"Material {db_item.material_name} (ID: {item_id}) in Project '{project_name}' is below reorder point ({db_item.reorder_point}). Current quantity: {quantity}."
#                 ))
#         try:
#             db.commit()
#             db.refresh(db_item)
#             db.refresh(db_item, attribute_names=['project', 'supplier'])
#             return db_item
#         except Exception as e:
#             db.rollback()
#             print(f"Error committing inventory quantity update for item {item_id}: {e}")
#             raise
#     return None

# def update_inventory_item_details(db: Session, item_id: int, item_update: InventoryCreate) -> Optional[Inventory]:
#     db_item = get_inventory_item(db, item_id)
#     if not db_item:
#         return None

#     update_data = item_update.model_dump(exclude_unset=True, exclude={'quantity'})

#     if "project_id" in update_data and update_data["project_id"] != db_item.project_id:
#         if not get_project_by_id(db, update_data["project_id"]):
#             raise ValueError(f"Project with ID {update_data['project_id']} not found.")
#     if "supplier_id" in update_data and update_data["supplier_id"] != db_item.supplier_id:
#          if update_data["supplier_id"] is not None and not get_supplier_by_id(db, update_data["supplier_id"]):
#              raise ValueError(f"Supplier with ID {update_data['supplier_id']} not found.")

#     new_name = update_data.get("material_name", db_item.material_name)
#     new_project_id = update_data.get("project_id", db_item.project_id)
#     if new_name != db_item.material_name or new_project_id != db_item.project_id:
#         existing_item = db.query(Inventory).filter(
#             Inventory.material_name == new_name,
#             Inventory.project_id == new_project_id,
#             Inventory.id != item_id
#         ).first()
#         if existing_item:
#             raise IntegrityError(f"Inventory item '{new_name}' already exists for Project ID {new_project_id}.", params=None, orig=None)

#     for key, value in update_data.items():
#         setattr(db_item, key, value)
#     db_item.last_updated = datetime.utcnow()

#     try:
#         db.commit()
#         db.refresh(db_item)
#         db.refresh(db_item, attribute_names=['project', 'supplier'])
#         return db_item
#     except IntegrityError as e:
#         db.rollback()
#         print(f"IntegrityError updating inventory item details {item_id}: {e}")
#         raise IntegrityError(f"Database error updating inventory item '{new_name}'.", params=None, orig=e)
#     except Exception as e:
#         db.rollback()
#         print(f"Unexpected error updating inventory item details {item_id}: {e}")
#         raise

# def delete_inventory_item(db: Session, item_id: int) -> bool:
#     db_item = get_inventory_item(db, item_id)
#     if db_item:
#         db.delete(db_item)
#         db.commit()
#         return True
#     return False

# # --- Consumption CRUD Operations (UPDATED to infer project_id) ---
# def get_consumption_data(db: Session, project_id: Optional[int] = None) -> List[Consumption]:
#     query = db.query(Consumption).options(
#         joinedload(Consumption.material),
#         joinedload(Consumption.project_rel)
#     )
#     if project_id is not None:
#         query = query.filter(Consumption.project_id == project_id)
#     return query.order_by(Consumption.date_used.desc()).all()


# def create_consumption_record(db: Session, consumption: ConsumptionCreate) -> Consumption:
#     """
#     Creates a consumption record. Behavior:
#       - If consumption.project_id is provided, it must match the inventory item's project_id.
#       - If consumption.project_id is omitted, the function will infer project_id from the inventory item.
#     """
#     # Retrieve the inventory item (independent of project)
#     inventory_item = get_inventory_item(db, consumption.material_id)
#     if not inventory_item:
#         raise ValueError(f"Material with ID {consumption.material_id} not found in inventory.")

#     # Determine project id (either from payload or inferred from inventory)
#     if consumption.project_id is None:
#         project_id_to_use = inventory_item.project_id
#     else:
#         project_id_to_use = consumption.project_id
#         # If provided project_id conflicts with material's project, raise clear error
#         if inventory_item.project_id != project_id_to_use:
#             raise ValueError(
#                 f"Material ID {consumption.material_id} belongs to Project ID {inventory_item.project_id}. "
#                 f"Provided project_id {project_id_to_use} does not match. "
#                 f"Either omit project_id to use the material's project or provide the correct project ID."
#             )

#     # Confirm project exists
#     project = get_project_by_id(db, project_id_to_use)
#     if not project:
#         raise ValueError(f"Project with ID {project_id_to_use} not found.")

#     # Ensure sufficient stock
#     if inventory_item.quantity < consumption.quantity_used:
#          raise ValueError(
#              f"Insufficient inventory for material '{inventory_item.material_name}' in Project '{project.name}'. "
#              f"Available: {inventory_item.quantity}, Needed: {consumption.quantity_used}"
#          )

#     # Create consumption record linking to determined project_id
#     db_consumption = Consumption(
#         material_id=consumption.material_id,
#         project_id=project_id_to_use,
#         quantity_used=consumption.quantity_used,
#         date_used=consumption.date_used or datetime.utcnow(),
#         notes=consumption.notes
#     )
#     db.add(db_consumption)

#     # Update inventory quantity
#     new_quantity = inventory_item.quantity - consumption.quantity_used
#     inventory_item.quantity = new_quantity
#     inventory_item.last_updated = datetime.utcnow()

#     try:
#         db.commit()
#         # Create low-stock alert if needed
#         if new_quantity <= inventory_item.reorder_point:
#              existing_alert = db.query(Alert).filter(Alert.material_id == inventory_item.id, Alert.alert_type == "low_stock", Alert.is_active == True).first()
#              if not existing_alert:
#                  create_alert(db, AlertCreate(
#                      material_id=inventory_item.id,
#                      alert_type="low_stock",
#                      message=f"Material {inventory_item.material_name} (ID: {inventory_item.id}) in Project '{project.name}' is below reorder point ({inventory_item.reorder_point}). Current quantity: {new_quantity}."
#                  ))

#         db.refresh(db_consumption)
#         db.refresh(db_consumption, attribute_names=['material', 'project_rel'])
#         return db_consumption
#     except Exception as e:
#         db.rollback()
#         print(f"Error during consumption creation/inventory update commit: {e}")
#         raise

# # --- Waste CRUD Operations (unchanged) ---
# def get_waste_data(db: Session, project_id: Optional[int] = None) -> List[Waste]:
#     query = db.query(Waste).options(
#         joinedload(Waste.material),
#         joinedload(Waste.project_rel)
#     )
#     if project_id is not None:
#         query = query.filter(Waste.project_id == project_id)
#     return query.order_by(Waste.date_recorded.desc()).all()


# def create_waste_record(db: Session, waste: WasteCreate) -> Waste:
#     project = get_project_by_id(db, waste.project_id)
#     if not project:
#          raise ValueError(f"Project with ID {waste.project_id} not found.")

#     inventory_item = db.query(Inventory).filter(
#         Inventory.id == waste.material_id,
#         Inventory.project_id == waste.project_id
#     ).first()
#     if not inventory_item:
#         raise ValueError(f"Material with ID {waste.material_id} not found in Project ID {waste.project_id}.")

#     db_waste = Waste(
#         material_id=waste.material_id,
#         project_id=waste.project_id,
#         quantity_wasted=waste.quantity_wasted,
#         reason=waste.reason,
#         preventive_measures=waste.preventive_measures,
#         date_recorded=datetime.utcnow()
#     )
#     db.add(db_waste)

#     new_quantity = inventory_item.quantity - waste.quantity_wasted
#     if new_quantity < 0:
#         print(f"Warning: Logging waste for material ID {waste.material_id} in project '{project.name}' resulted in negative stock. Setting stock to 0.")
#         new_quantity = 0

#     inventory_item.quantity = new_quantity
#     inventory_item.last_updated = datetime.utcnow()

#     try:
#         db.commit()
#         if new_quantity <= inventory_item.reorder_point:
#              existing_alert = db.query(Alert).filter(Alert.material_id == inventory_item.id, Alert.alert_type == "low_stock", Alert.is_active == True).first()
#              if not existing_alert:
#                  create_alert(db, AlertCreate( material_id=inventory_item.id, alert_type="low_stock", message=f"Material {inventory_item.material_name} (ID: {inventory_item.id}) in Project '{project.name}' is below reorder point ({inventory_item.reorder_point}). Current quantity: {new_quantity}." ))

#         db.refresh(db_waste)
#         db.refresh(db_waste, attribute_names=['material', 'project_rel'])
#         return db_waste
#     except Exception as e:
#         db.rollback()
#         print(f"Error during waste creation/inventory update commit: {e}")
#         raise

# def get_waste_record(db: Session, waste_id: int) -> Optional[Waste]:
#     return db.query(Waste).options(
#         joinedload(Waste.material),
#         joinedload(Waste.project_rel)
#     ).filter(Waste.id == waste_id).first()

# # --- Cost CRUD Operations (unchanged) ---
# def get_cost_data(db: Session) -> List[Cost]:
#     return db.query(Cost).options(
#         joinedload(Cost.material),
#         joinedload(Cost.supplier)
#     ).order_by(Cost.date_recorded.desc()).all()

# def create_cost_record(db: Session, cost: CostCreate) -> Cost:
#     cost_data = cost.model_dump()
#     inventory_item = get_inventory_item(db, cost.material_id)
#     if not inventory_item:
#          raise ValueError(f"Material with ID {cost.material_id} not found in inventory.")
#     if cost.supplier_id and not get_supplier_by_id(db, cost.supplier_id):
#          raise ValueError(f"Supplier with ID {cost.supplier_id} not found.")

#     if cost_data.get("total_cost") is None:
#         unit_price = cost_data.get("unit_price")
#         quantity_purchased = cost_data.get("quantity_purchased")
#         if unit_price is not None and quantity_purchased is not None:
#              cost_data["total_cost"] = unit_price * quantity_purchased
#         else:
#              missing_fields = []
#              if unit_price is None: missing_fields.append("unit_price")
#              if quantity_purchased is None: missing_fields.append("quantity_purchased")
#              raise ValueError(f"Cannot calculate total_cost: Required fields missing: {', '.join(missing_fields)}")
#     elif cost_data["total_cost"] < 0:
#          raise ValueError("Total cost cannot be negative.")

#     cost_data.setdefault("date_recorded", datetime.utcnow())

#     if isinstance(cost_data.get("date_recorded"), str):
#         try:
#             from dateutil import parser
#             cost_data["date_recorded"] = parser.parse(cost_data["date_recorded"])
#         except (ImportError, ValueError):
#              print(f"Warning: Could not parse date string '{cost_data.get('date_recorded')}'. Using current UTC time.")
#              cost_data["date_recorded"] = datetime.utcnow()

#     db_cost = Cost(**cost_data)
#     db.add(db_cost)
#     try:
#         db.commit()
#         db.refresh(db_cost)
#         db.refresh(db_cost, attribute_names=['material', 'supplier'])
#         return db_cost
#     except IntegrityError as e:
#         db.rollback()
#         if "NOT NULL constraint failed: costs.total_cost" in str(e.orig):
#              print(f"IntegrityError: total_cost was unexpectedly NULL before commit. Data: {cost_data}")
#         else:
#              print(f"IntegrityError creating cost record: {e}")
#         raise
#     except Exception as e:
#         db.rollback()
#         print(f"Unexpected error creating cost record: {e}")
#         raise

# # --- Alerts (unchanged) ---
# def get_alerts(db: Session) -> List[Alert]:
#     return db.query(Alert).options(
#         joinedload(Alert.material).joinedload(Inventory.project)
#         ).filter(Alert.is_active == True).order_by(Alert.date_created.desc()).all()

# def create_alert(db: Session, alert: AlertCreate) -> Alert:
#     existing_alert = db.query(Alert).filter(
#         Alert.material_id == alert.material_id,
#         Alert.alert_type == alert.alert_type,
#         Alert.is_active == True
#     ).first()
#     if existing_alert:
#         print(f"Skipping duplicate active alert creation for material ID {alert.material_id}, type {alert.alert_type}")
#         return existing_alert

#     db_alert = Alert(**alert.model_dump())
#     db.add(db_alert)
#     try:
#         db.commit()
#         db.refresh(db_alert)
#         return db_alert
#     except Exception as e:
#          db.rollback()
#          print(f"Error creating alert: {e}")
#          return None

# def resolve_alert(db: Session, alert_id: int) -> Optional[Alert]:
#     db_alert = db.query(Alert).filter(Alert.id == alert_id).first()
#     if db_alert:
#         db_alert.is_active = False
#         db.commit()
#         db.refresh(db_alert)
#         return db_alert
#     return None
# crud.py
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func
from datetime import datetime
from typing import List, Optional

# Import your models and schemas
from models import Inventory, Supplier, Consumption, Cost, Waste, Alert, Project
from schemas import (
    InventoryCreate, InventoryRead,
    SupplierCreate, SupplierRead, SupplierUpdate,
    ConsumptionCreate, ConsumptionRead,
    CostCreate, CostRead,
    WasteCreate, WasteRead,
    AlertCreate, AlertRead,
    ProjectCreate, ProjectUpdate, ProjectRead
)

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
        raise IntegrityError(f"Database error updating project.", params=None, orig=e)
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
                Alert.is_active == True
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
             existing_alert = db.query(Alert).filter(Alert.material_id == inventory_item.id, Alert.alert_type == "low_stock", Alert.is_active == True).first()
             if not existing_alert:
                 create_alert(db, AlertCreate( material_id=inventory_item.id, alert_type="low_stock", message=f"Material {inventory_item.material_name} (ID: {inventory_item.id}) in Project '{project.name}' is below reorder point ({inventory_item.reorder_point}). Current quantity: {new_quantity}." ))
        # check limit exceeded alert
        db.refresh(inventory_item)
        check_and_create_limit_alert(db, inventory_item)
        db.refresh(db_consumption)
        db.refresh(db_consumption, attribute_names=['material', 'project_rel'])
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
             existing_alert = db.query(Alert).filter(Alert.material_id == inventory_item.id, Alert.alert_type == "low_stock", Alert.is_active == True).first()
             if not existing_alert:
                 create_alert(db, AlertCreate( material_id=inventory_item.id, alert_type="low_stock", message=f"Material {inventory_item.material_name} (ID: {inventory_item.id}) in Project '{project.name}' is below reorder point ({inventory_item.reorder_point}). Current quantity: {new_quantity}." ))
        # check limit exceeded alert
        db.refresh(inventory_item)
        check_and_create_limit_alert(db, inventory_item)
        db.refresh(db_waste)
        db.refresh(db_waste, attribute_names=['material', 'project_rel'])
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
             if unit_price is None: missing_fields.append("unit_price")
             if quantity_purchased is None: missing_fields.append("quantity_purchased")
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
        ).filter(Alert.is_active == True).order_by(Alert.date_created.desc()).all()

def create_alert(db: Session, alert: AlertCreate) -> Alert:
    existing_alert = db.query(Alert).filter(
        Alert.material_id == alert.material_id,
        Alert.alert_type == alert.alert_type,
        Alert.is_active == True
    ).first()
    if existing_alert:
        print(f"Skipping duplicate active alert creation for material ID {alert.material_id}, type {alert.alert_type}")
        return existing_alert
    db_alert = Alert(**alert.model_dump())
    db.add(db_alert)
    try:
        db.commit()
        db.refresh(db_alert)
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
            Alert.is_active == True
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
        existing = db.query(Alert).filter(Alert.material_id == inventory_item.id, Alert.alert_type == "limit_exceeded", Alert.is_active == True).first()
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
