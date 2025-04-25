# schemas.py

from pydantic import BaseModel, Field # Ensure Field is imported
from datetime import datetime, date # Added date
from typing import Optional, List # Added List if needed elsewhere

# --- Inventory Schemas (Updated) ---
class InventoryBase(BaseModel):
    material_name: str = Field(..., min_length=1)
    quantity: float = Field(..., ge=0) # Quantity should not be negative
    unit: str = Field(..., min_length=1)
    reorder_point: float = Field(..., ge=0) # Reorder point should not be negative
    supplier_id: Optional[int] = None # Supplier might be optional
    project_id: int # Link to project (Required as per models.py)

class InventoryCreate(InventoryBase):
    # project_id is required via InventoryBase
    pass

class InventoryRead(InventoryBase):
    id: int
    last_updated: Optional[datetime] = None
    # Include related names for convenience
    supplier_name: Optional[str] = None
    project_name: Optional[str] = None # Populated from project.name

    class Config:
        from_attributes = True # Enable ORM mode for Pydantic V2

# --- Supplier Schemas ---
class SupplierBase(BaseModel):
    name: str
    contact_person: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    lead_time_days: Optional[int] = Field(None, gt=0) # If provided, must be > 0
    reliability_rating: Optional[float] = Field(None, ge=0, le=5) # Rating 0-5

class SupplierCreate(SupplierBase):
    name: str = Field(..., min_length=1)
    # Make other fields required for creation if necessary
    contact_person: Optional[str] = None # Keep optional based on user's model
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    lead_time_days: Optional[int] = Field(None, gt=0)
    reliability_rating: Optional[float] = Field(None, ge=0, le=5)

class SupplierUpdate(SupplierBase):
    # All fields are optional during update
    pass

class SupplierRead(SupplierBase):
    id: int
    class Config:
        from_attributes = True

# --- Consumption Schemas (Updated) ---
class ConsumptionBase(BaseModel):
    material_id: int
    project_id: int # Link to Project ID (Required as per models.py)
    quantity_used: float = Field(..., gt=0) # Must consume positive amount
    notes: Optional[str] = None
    # Removed old 'project: str' field

class ConsumptionCreate(ConsumptionBase):
    # date_used defaults in the model or can be set here if needed
    date_used: Optional[datetime] = Field(default_factory=datetime.utcnow)

class ConsumptionRead(ConsumptionBase):
    id: int
    date_used: datetime
    # Add related names
    material_name: Optional[str] = None # Populated from material relationship
    project_name: Optional[str] = None # Populated from project_rel.name

    class Config:
        from_attributes = True

# --- Cost Schemas ---
class CostBase(BaseModel):
    material_id: int
    supplier_id: Optional[int] = None # Allow optional supplier
    unit_price: float = Field(..., ge=0)
    quantity_purchased: float = Field(..., gt=0)
    total_cost: Optional[float] = Field(None, ge=0) # Calculated if None
    date_recorded: Optional[datetime] = None # Defaults in model/CRUD
    notes: Optional[str] = None

class CostCreate(CostBase):
     # Make supplier required for creation if needed, else keep optional from Base
     # supplier_id: int
     pass # Inherits optional supplier_id from Base

class CostRead(CostBase):
    id: int
    date_recorded: datetime # Ensure date is included
    # Add related names
    material_name: Optional[str] = None
    supplier_name: Optional[str] = None

    class Config:
        from_attributes = True

# --- Waste Schemas (Updated) ---
class WasteBase(BaseModel):
    material_id: int
    project_id: int # Link to Project ID (Required as per models.py)
    quantity_wasted: float = Field(..., gt=0) # Must waste positive amount
    reason: str = Field(..., min_length=1)
    preventive_measures: Optional[str] = None
    # Removed old 'project_name: str' field

class WasteCreate(WasteBase):
    # date_recorded defaults in the model
    pass

class WasteRead(WasteBase):
    id: int
    date_recorded: datetime
    # Add related names
    material_name: Optional[str] = None # Populated from material relationship
    project_name: Optional[str] = None # Populated from project_rel.name

    class Config:
        from_attributes = True

# --- Alert Schemas ---
class AlertBase(BaseModel):
    material_id: int
    alert_type: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    is_active: bool = True # Use bool

class AlertCreate(AlertBase):
    pass

class AlertRead(AlertBase):
    id: int
    date_created: datetime

    class Config:
        from_attributes = True

# --- Project Schemas ---
class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, description="Name of the construction project")
    location: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: Optional[str] = "Planning"

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(ProjectBase):
    # All fields optional on update, but if name is provided, it must be valid
    name: Optional[str] = Field(None, min_length=1, description="New name for the project")

class ProjectRead(ProjectBase):
    id: int
    class Config:
        from_attributes = True

