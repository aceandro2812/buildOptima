from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# INVENTORY SCHEMAS
class InventoryBase(BaseModel):
    material_name: str
    quantity: float
    unit: str
    reorder_point: float
    supplier_id: int

class InventoryCreate(InventoryBase):
    pass

class InventoryRead(InventoryBase):
    id: int
    last_updated: datetime

    class Config:
        from_attributes = True

# SUPPLIER SCHEMAS
class SupplierBase(BaseModel):
    name: str
    contact_person: str
    email: str
    phone: str
    address: str
    lead_time_days: int
    reliability_rating: float

class SupplierCreate(SupplierBase):
    pass

class SupplierRead(SupplierBase):
    id: int

    class Config:
        from_attributes = True

# CONSUMPTION SCHEMAS
class ConsumptionBase(BaseModel):
    material_id: int
    project: str      # Changed from project_name to project
    quantity_used: float
    notes: Optional[str] = None

class ConsumptionCreate(ConsumptionBase):
    date_used: datetime = datetime.utcnow()  # Default if not provided

class ConsumptionRead(ConsumptionBase):
    id: int
    date_used: datetime

    class Config:
        from_attributes = True

# COST SCHEMAS
class CostBase(BaseModel):
    material_id: int
    supplier_id: int
    unit_price: float
    quantity_purchased: float
    total_cost: Optional[float] = None   # Will be computed if not provided
    date_recorded: Optional[datetime] = None
    notes: Optional[str] = None

class CostCreate(CostBase):
    pass

class CostRead(CostBase):
    id: int

    class Config:
        from_attributes = True

# WASTE SCHEMAS
class WasteBase(BaseModel):
    material_id: int
    project_name: str
    quantity_wasted: float
    reason: str
    preventive_measures: Optional[str] = None

class WasteCreate(WasteBase):
    pass

class WasteRead(WasteBase):
    id: int
    date_recorded: datetime

    class Config:
        from_attributes = True

# ALERT SCHEMAS
class AlertBase(BaseModel):
    material_id: int
    alert_type: str
    message: str
    is_active: int = 1

class AlertCreate(AlertBase):
    pass

class AlertRead(AlertBase):
    id: int
    date_created: datetime

    class Config:
        from_attributes = True
