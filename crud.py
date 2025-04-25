# crud.py

from sqlalchemy.orm import Session, joinedload # Import joinedload
from sqlalchemy.exc import IntegrityError
from datetime import datetime
from typing import List, Optional

# Import Models and Schemas
from models import Inventory, Supplier, Consumption, Cost, Waste, Alert, Project
from schemas import ( # Import specific schemas needed
    InventoryCreate, InventoryRead, # InventoryUpdate (if created)
    SupplierCreate, SupplierRead, SupplierUpdate,
    ConsumptionCreate, ConsumptionRead,
    CostCreate, CostRead,
    WasteCreate, WasteRead,
    AlertCreate, AlertRead,
    ProjectCreate, ProjectUpdate, ProjectRead
)

# --- Project CRUD Operations ---
# (Placed first as they are used for validation below)

def get_project_by_id(db: Session, project_id: int) -> Optional[Project]:
    """Gets a single project by its ID."""
    return db.query(Project).filter(Project.id == project_id).first()

def get_project_by_name(db: Session, name: str) -> Optional[Project]:
    """Gets a single project by its unique name."""
    return db.query(Project).filter(Project.name == name).first()

def get_projects(db: Session, skip: int = 0, limit: int = 100) -> List[Project]:
    """Gets a list of projects with pagination."""
    return db.query(Project).order_by(Project.name).offset(skip).limit(limit).all()

def create_project(db: Session, project: ProjectCreate) -> Project:
    """Creates a new project."""
    existing_project = get_project_by_name(db, project.name)
    if existing_project:
        # Raise specific error for uniqueness constraint
        raise IntegrityError(f"Project with name '{project.name}' already exists.", params=None, orig=None)

    db_project = Project(**project.model_dump())
    db.add(db_project)
    try:
        db.commit()
        db.refresh(db_project)
        return db_project
    except IntegrityError as e:
        db.rollback() # Rollback the session in case of commit error
        print(f"IntegrityError creating project: {e}") # Log the error
        # Re-raise a more specific error or handle as needed
        raise IntegrityError(f"Database error creating project '{project.name}'. It might already exist or another constraint failed.", params=None, orig=e)
    except Exception as e:
        db.rollback()
        print(f"Unexpected error creating project: {e}")
        raise # Re-raise other exceptions


def update_project(db: Session, project_id: int, project_update: ProjectUpdate) -> Optional[Project]:
    """Updates an existing project."""
    db_project = get_project_by_id(db, project_id)
    if not db_project:
        return None # Project not found

    update_data = project_update.model_dump(exclude_unset=True) # Only update provided fields

    # Check for name uniqueness if name is being updated to a different value
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
        raise IntegrityError(f"Database error updating project '{update_data.get('name', db_project.name)}'. Name might conflict or another constraint failed.", params=None, orig=e)
    except Exception as e:
        db.rollback()
        print(f"Unexpected error updating project: {e}")
        raise


def delete_project(db: Session, project_id: int) -> bool:
    """
    Deletes a project by its ID.
    Raises ValueError if the project has linked records (Inventory, Consumption, Waste).
    """
    db_project = get_project_by_id(db, project_id)
    if db_project:
        # Check for linked records before deleting (safer than relying solely on cascade)
        if db_project.inventory_items:
            raise ValueError(f"Cannot delete project ID {project_id} as it has linked inventory items.")
        if db_project.consumption_records:
            raise ValueError(f"Cannot delete project ID {project_id} as it has linked consumption records.")
        if db_project.waste_records:
            raise ValueError(f"Cannot delete project ID {project_id} as it has linked waste records.")

        db.delete(db_project)
        db.commit()
        return True
    return False # Project not found

# --- Supplier CRUD Operations ---
# (Placed here as needed for Inventory validation)

def get_suppliers(db: Session) -> List[Supplier]:
    """Gets all suppliers."""
    return db.query(Supplier).all()

def create_supplier_db(db: Session, supplier: SupplierCreate) -> Supplier:
    """Creates a new supplier."""
    db_supplier = Supplier(**supplier.model_dump())
    db.add(db_supplier)
    db.commit()
    db.refresh(db_supplier)
    return db_supplier

def get_supplier_by_id(db: Session, supplier_id: int) -> Optional[Supplier]:
    """Gets a supplier by ID."""
    return db.query(Supplier).filter(Supplier.id == supplier_id).first()

def update_supplier_db(db: Session, supplier_id: int, supplier: SupplierUpdate) -> Optional[Supplier]:
    """Updates an existing supplier."""
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
    """
    Deletes a supplier by ID.
    Raises ValueError if the supplier is linked to inventory items.
    """
    db_supplier = get_supplier_by_id(db, supplier_id)
    if db_supplier:
        # Check if supplier is linked to materials before deleting
        linked_materials = db.query(Inventory).filter(Inventory.supplier_id == supplier_id).first()
        if linked_materials:
             raise ValueError(f"Cannot delete supplier ID {supplier_id} as it is linked to materials.")
        db.delete(db_supplier)
        db.commit()
        return True
    return False

# --- Inventory CRUD Operations (Updated) ---

def get_inventory_item(db: Session, item_id: int) -> Optional[Inventory]:
    """Gets a single inventory item by ID, eager loading relationships."""
    # This function is used internally by others, ensure it loads needed relationships
    return db.query(Inventory).options(
        joinedload(Inventory.supplier),
        joinedload(Inventory.project)
    ).filter(Inventory.id == item_id).first()

def get_inventory(db: Session, project_id: Optional[int] = None) -> List[Inventory]:
    """
    Gets all inventory items, optionally filtered by project_id.
    Eager loads supplier and project.
    """
    query = db.query(Inventory).options(
        joinedload(Inventory.supplier),
        joinedload(Inventory.project)
    )
    if project_id is not None:
        query = query.filter(Inventory.project_id == project_id)
    return query.order_by(Inventory.project_id, Inventory.material_name).all()

def create_inventory(db: Session, item: InventoryCreate) -> Inventory:
    """Creates an inventory item, ensuring the linked project exists."""
    # Check if project exists
    project = get_project_by_id(db, item.project_id)
    if not project:
        raise ValueError(f"Project with ID {item.project_id} not found.")
    # Check if supplier exists if provided
    if item.supplier_id:
        supplier = get_supplier_by_id(db, item.supplier_id)
        if not supplier:
             raise ValueError(f"Supplier with ID {item.supplier_id} not found.")

    # Check if material with the same name already exists within the same project
    existing_item = db.query(Inventory).filter(
        Inventory.material_name == item.material_name,
        Inventory.project_id == item.project_id
    ).first()
    if existing_item:
        raise IntegrityError(f"Inventory item '{item.material_name}' already exists for Project ID {item.project_id}.", params=None, orig=None)


    db_item = Inventory(**item.model_dump())
    db.add(db_item)
    try:
        db.commit()
        db.refresh(db_item)
        # Refresh relationships to load them for the return object
        db.refresh(db_item, attribute_names=['project', 'supplier'])
        return db_item
    except IntegrityError as e:
        db.rollback()
        print(f"IntegrityError creating inventory item: {e}")
        raise IntegrityError(f"Database error creating inventory item '{item.material_name}'. Name might conflict within the project.", params=None, orig=e)
    except Exception as e:
        db.rollback()
        print(f"Unexpected error creating inventory item: {e}")
        raise


def update_inventory_quantity(db: Session, item_id: int, quantity: float) -> Optional[Inventory]:
    """
    Updates only the quantity of an inventory item and checks for low stock alerts.
    Used internally by consumption/waste logging.
    """
    db_item = get_inventory_item(db, item_id) # Use getter to ensure relationships are loaded
    if db_item:
        db_item.quantity = quantity
        db_item.last_updated = datetime.utcnow()
        # Check for alert *after* updating quantity
        if quantity <= db_item.reorder_point:
            existing_alert = db.query(Alert).filter(
                Alert.material_id == item_id,
                Alert.alert_type == "low_stock",
                Alert.is_active == True
            ).first()
            if not existing_alert:
                # Ensure project name is available for the alert message
                project_name = db_item.project.name if db_item.project else "Unknown Project"
                create_alert(db, AlertCreate(
                    material_id=item_id,
                    alert_type="low_stock",
                    message=f"Material {db_item.material_name} (ID: {item_id}) in Project '{project_name}' is below reorder point ({db_item.reorder_point}). Current quantity: {quantity}."
                ))
        # Commit the quantity change here (or rely on caller to commit if part of larger transaction)
        # For simplicity here, we commit.
        try:
            db.commit()
            db.refresh(db_item)
            db.refresh(db_item, attribute_names=['project', 'supplier']) # Refresh relationships
            return db_item
        except Exception as e:
            db.rollback()
            print(f"Error committing inventory quantity update for item {item_id}: {e}")
            raise # Re-raise error
    return None

def update_inventory_item_details(db: Session, item_id: int, item_update: InventoryCreate) -> Optional[Inventory]:
    """
    Updates details (name, unit, reorder point, supplier, project) of an inventory item.
    Does NOT update quantity directly (use update_inventory_quantity for that).
    Note: Using InventoryCreate schema for update payload. Consider an InventoryUpdate schema.
    """
    db_item = get_inventory_item(db, item_id)
    if not db_item:
        return None

    update_data = item_update.model_dump(exclude_unset=True, exclude={'quantity'}) # Exclude quantity

    # Validate foreign keys if they are being updated
    if "project_id" in update_data and update_data["project_id"] != db_item.project_id:
        if not get_project_by_id(db, update_data["project_id"]):
            raise ValueError(f"Project with ID {update_data['project_id']} not found.")
    if "supplier_id" in update_data and update_data["supplier_id"] != db_item.supplier_id:
         if update_data["supplier_id"] is not None and not get_supplier_by_id(db, update_data["supplier_id"]):
             raise ValueError(f"Supplier with ID {update_data['supplier_id']} not found.")

    # Check for name conflict if name or project_id is changing
    new_name = update_data.get("material_name", db_item.material_name)
    new_project_id = update_data.get("project_id", db_item.project_id)
    if new_name != db_item.material_name or new_project_id != db_item.project_id:
        existing_item = db.query(Inventory).filter(
            Inventory.material_name == new_name,
            Inventory.project_id == new_project_id,
            Inventory.id != item_id # Exclude self
        ).first()
        if existing_item:
            raise IntegrityError(f"Inventory item '{new_name}' already exists for Project ID {new_project_id}.", params=None, orig=None)

    for key, value in update_data.items():
        setattr(db_item, key, value)
    db_item.last_updated = datetime.utcnow() # Update timestamp even if only details change

    try:
        db.commit()
        db.refresh(db_item)
        db.refresh(db_item, attribute_names=['project', 'supplier']) # Refresh relationships
        return db_item
    except IntegrityError as e:
        db.rollback()
        print(f"IntegrityError updating inventory item details {item_id}: {e}")
        raise IntegrityError(f"Database error updating inventory item '{new_name}'. Name might conflict within the project.", params=None, orig=e)
    except Exception as e:
        db.rollback()
        print(f"Unexpected error updating inventory item details {item_id}: {e}")
        raise

def delete_inventory_item(db: Session, item_id: int) -> bool:
    """Deletes an inventory item. Cascade delete handles related records if configured."""
    db_item = get_inventory_item(db, item_id)
    if db_item:
        db.delete(db_item)
        db.commit()
        return True
    return False

# --- Consumption CRUD Operations (Updated) ---
def get_consumption_data(db: Session, project_id: Optional[int] = None) -> List[Consumption]:
    """
    Gets consumption data, optionally filtered by project_id.
    Eager loads material and project.
    """
    query = db.query(Consumption).options(
        joinedload(Consumption.material),#.joinedload(Inventory.supplier), # Optionally load supplier via material
        joinedload(Consumption.project_rel)
    )
    if project_id is not None:
        query = query.filter(Consumption.project_id == project_id)
    return query.order_by(Consumption.date_used.desc()).all()


def create_consumption_record(db: Session, consumption: ConsumptionCreate) -> Consumption:
    """Creates a consumption record, linking to a project via project_id."""
    # Check if project exists
    project = get_project_by_id(db, consumption.project_id)
    if not project:
         raise ValueError(f"Project with ID {consumption.project_id} not found.")

    # Find the specific inventory item for this material within this project
    inventory_item = db.query(Inventory).filter(
        Inventory.id == consumption.material_id,
        Inventory.project_id == consumption.project_id
    ).first()

    if not inventory_item:
        raise ValueError(f"Material with ID {consumption.material_id} not found in Project ID {consumption.project_id}.")
    if inventory_item.quantity < consumption.quantity_used:
         raise ValueError(f"Insufficient inventory for material '{inventory_item.material_name}' in Project '{project.name}'. Available: {inventory_item.quantity}, Needed: {consumption.quantity_used}")

    # Create the DB object using project_id
    db_consumption = Consumption(
        material_id=consumption.material_id,
        project_id=consumption.project_id,
        quantity_used=consumption.quantity_used,
        date_used=consumption.date_used or datetime.utcnow(),
        notes=consumption.notes
    )
    db.add(db_consumption)

    # Update inventory quantity
    new_quantity = inventory_item.quantity - consumption.quantity_used
    inventory_item.quantity = new_quantity
    inventory_item.last_updated = datetime.utcnow()

    try:
        db.commit() # Commit consumption record and inventory update together
        # Check for alert *after* successful commit
        if new_quantity <= inventory_item.reorder_point:
             existing_alert = db.query(Alert).filter(Alert.material_id == inventory_item.id, Alert.alert_type == "low_stock", Alert.is_active == True).first()
             if not existing_alert:
                 # Create alert in a separate transaction or handle potential nested commit issues
                 # For simplicity, we'll assume create_alert commits separately if needed
                 create_alert(db, AlertCreate( material_id=inventory_item.id, alert_type="low_stock", message=f"Material {inventory_item.material_name} (ID: {inventory_item.id}) in Project '{project.name}' is below reorder point ({inventory_item.reorder_point}). Current quantity: {new_quantity}." ))

        db.refresh(db_consumption)
        db.refresh(db_consumption, attribute_names=['material', 'project_rel']) # Refresh relationships
        return db_consumption
    except Exception as e:
        db.rollback() # Rollback if commit fails
        print(f"Error during consumption creation/inventory update commit: {e}")
        raise


# --- Waste CRUD Operations (Updated) ---
def get_waste_data(db: Session, project_id: Optional[int] = None) -> List[Waste]:
    """
    Gets waste data, optionally filtered by project_id.
    Eager loads material and project.
    """
    query = db.query(Waste).options(
        joinedload(Waste.material),
        joinedload(Waste.project_rel)
    )
    if project_id is not None:
        query = query.filter(Waste.project_id == project_id)
    return query.order_by(Waste.date_recorded.desc()).all()


def create_waste_record(db: Session, waste: WasteCreate) -> Waste:
    """Creates a waste record, linking to a project via project_id."""
    # Check if project exists
    project = get_project_by_id(db, waste.project_id)
    if not project:
         raise ValueError(f"Project with ID {waste.project_id} not found.")

    # Check inventory exists within the specified project
    inventory_item = db.query(Inventory).filter(
        Inventory.id == waste.material_id,
        Inventory.project_id == waste.project_id
    ).first()
    if not inventory_item:
        raise ValueError(f"Material with ID {waste.material_id} not found in Project ID {waste.project_id}.")

    # Create the DB object using project_id
    db_waste = Waste(
        material_id=waste.material_id,
        project_id=waste.project_id,
        quantity_wasted=waste.quantity_wasted,
        reason=waste.reason,
        preventive_measures=waste.preventive_measures,
        date_recorded=datetime.utcnow()
    )
    db.add(db_waste)

    # Update inventory based on waste
    new_quantity = inventory_item.quantity - waste.quantity_wasted
    if new_quantity < 0:
        print(f"Warning: Logging waste for material ID {waste.material_id} in project '{project.name}' resulted in negative stock. Setting stock to 0.")
        new_quantity = 0

    # Update the specific project's inventory item
    inventory_item.quantity = new_quantity
    inventory_item.last_updated = datetime.utcnow()

    try:
        db.commit() # Commit waste record and inventory update together
        # Check for alert *after* successful commit
        if new_quantity <= inventory_item.reorder_point:
             existing_alert = db.query(Alert).filter(Alert.material_id == inventory_item.id, Alert.alert_type == "low_stock", Alert.is_active == True).first()
             if not existing_alert:
                 create_alert(db, AlertCreate( material_id=inventory_item.id, alert_type="low_stock", message=f"Material {inventory_item.material_name} (ID: {inventory_item.id}) in Project '{project.name}' is below reorder point ({inventory_item.reorder_point}). Current quantity: {new_quantity}." ))

        db.refresh(db_waste)
        db.refresh(db_waste, attribute_names=['material', 'project_rel']) # Refresh relationships
        return db_waste
    except Exception as e:
        db.rollback() # Rollback if commit fails
        print(f"Error during waste creation/inventory update commit: {e}")
        raise


def get_waste_record(db: Session, waste_id: int) -> Optional[Waste]:
    """Gets a single waste record by ID, eager loading relationships."""
    return db.query(Waste).options(
        joinedload(Waste.material),
        joinedload(Waste.project_rel)
    ).filter(Waste.id == waste_id).first()


# --- Cost CRUD Operations (FIXED) ---
def get_cost_data(db: Session) -> List[Cost]:
    """Gets all cost data, eager loading relationships."""
    return db.query(Cost).options(
        joinedload(Cost.material), # Load material
        joinedload(Cost.supplier)  # Load supplier
    ).order_by(Cost.date_recorded.desc()).all()

def create_cost_record(db: Session, cost: CostCreate) -> Cost:
    """Creates a cost record."""
    cost_data = cost.model_dump()
    # Validate foreign keys
    # Use the specific inventory item getter to ensure relationships are loaded if needed later
    inventory_item = get_inventory_item(db, cost.material_id)
    if not inventory_item:
         raise ValueError(f"Material with ID {cost.material_id} not found in inventory.")
    if cost.supplier_id and not get_supplier_by_id(db, cost.supplier_id):
         raise ValueError(f"Supplier with ID {cost.supplier_id} not found.")

    # --- FIX START: Calculate total_cost correctly ---
    # Check if total_cost is None or missing, and calculate if necessary
    if cost_data.get("total_cost") is None:
        unit_price = cost_data.get("unit_price")
        quantity_purchased = cost_data.get("quantity_purchased")

        if unit_price is not None and quantity_purchased is not None:
             cost_data["total_cost"] = unit_price * quantity_purchased
        else:
             # This case should ideally be prevented by Pydantic validation,
             # but added for robustness.
             missing_fields = []
             if unit_price is None: missing_fields.append("unit_price")
             if quantity_purchased is None: missing_fields.append("quantity_purchased")
             raise ValueError(f"Cannot calculate total_cost: Required fields missing: {', '.join(missing_fields)}")
    # Ensure total_cost is non-negative (Pydantic schema already does ge=0)
    elif cost_data["total_cost"] < 0:
         raise ValueError("Total cost cannot be negative.")
    # --- FIX END ---

    # Set default date if not provided
    # Use setdefault for date as it's truly optional in the payload
    cost_data.setdefault("date_recorded", datetime.utcnow())

    # Ensure date_recorded is a datetime object if provided (Pydantic handles this)
    if isinstance(cost_data.get("date_recorded"), str):
        try:
            # Pydantic v2 usually handles date parsing, but belt-and-suspenders
            from dateutil import parser
            cost_data["date_recorded"] = parser.parse(cost_data["date_recorded"])
        except (ImportError, ValueError):
             # Fallback or raise error if parsing fails and date is required
             print(f"Warning: Could not parse date string '{cost_data.get('date_recorded')}'. Using current UTC time.")
             cost_data["date_recorded"] = datetime.utcnow()


    db_cost = Cost(**cost_data) # Now total_cost should have a valid float value
    db.add(db_cost)
    try:
        db.commit() # Commit should now succeed
        db.refresh(db_cost)
        # Eagerly load relationships for the return object if needed by the API response
        db.refresh(db_cost, attribute_names=['material', 'supplier'])
        return db_cost
    except IntegrityError as e:
        db.rollback()
        # Provide more context if the error persists unexpectedly
        if "NOT NULL constraint failed: costs.total_cost" in str(e.orig):
             print(f"IntegrityError: total_cost was unexpectedly NULL before commit. Data: {cost_data}")
        else:
             print(f"IntegrityError creating cost record: {e}")
        raise # Re-raise the original exception for FastAPI to handle
    except Exception as e:
        db.rollback()
        print(f"Unexpected error creating cost record: {e}")
        raise

# --- Alert CRUD Operations ---
def get_alerts(db: Session) -> List[Alert]:
    """Gets all active alerts."""
    # Eager load material and project for context in the alert message if needed
    return db.query(Alert).options(
        joinedload(Alert.material).joinedload(Inventory.project)
        ).filter(Alert.is_active == True).order_by(Alert.date_created.desc()).all()

def create_alert(db: Session, alert: AlertCreate) -> Alert:
    """Creates an alert, preventing exact duplicate active alerts."""
    existing_alert = db.query(Alert).filter(
        Alert.material_id == alert.material_id,
        Alert.alert_type == alert.alert_type,
        Alert.is_active == True
    ).first()
    if existing_alert:
        # Optionally update timestamp or message if alert condition persists
        # existing_alert.date_created = datetime.utcnow()
        # existing_alert.message = alert.message # Update message with latest quantity?
        # db.commit()
        # db.refresh(existing_alert)
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
         # Decide how to handle alert creation failure (e.g., log, raise)
         # Raising might interfere with the primary operation (consumption/waste log)
         return None # Or re-raise if alert is critical


def resolve_alert(db: Session, alert_id: int) -> Optional[Alert]:
    """Marks an alert as inactive."""
    db_alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if db_alert:
        db_alert.is_active = False
        db.commit()
        db.refresh(db_alert)
        return db_alert
    return None