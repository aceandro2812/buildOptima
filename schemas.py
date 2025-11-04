# schemas.py

from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import Optional, List

# schemas.py (Inventory parts)

class InventoryBase(BaseModel):
    material_name: str = Field(..., min_length=1)
    quantity: float = Field(..., ge=0)
    unit: str = Field(..., min_length=1)
    reorder_point: float = Field(..., ge=0)
    supplier_id: Optional[int] = None
    project_id: int

    # New estimate fields (optional)
    estimated_quantity: Optional[float] = Field(None, ge=0)
    estimated_unit_price: Optional[float] = Field(None, ge=0)
    estimated_total_value: Optional[float] = Field(None, ge=0)

class InventoryCreate(InventoryBase):
    pass

class InventoryRead(InventoryBase):
    id: int
    last_updated: Optional[datetime] = None
    supplier_name: Optional[str] = None
    project_name: Optional[str] = None

    # Read-only computed fields (returned in report)
    actual_consumed_quantity: Optional[float] = None
    actual_value_used: Optional[float] = None
    variance_value: Optional[float] = None
    variance_percent: Optional[float] = None
    status: Optional[str] = None  # exceeded / under / no_estimate

    class Config:
        from_attributes = True

# --- Supplier Schemas ---
class SupplierBase(BaseModel):
    name: str
    contact_person: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    lead_time_days: Optional[int] = Field(None, gt=0)
    reliability_rating: Optional[float] = Field(None, ge=0, le=5)

class SupplierCreate(SupplierBase):
    name: str = Field(..., min_length=1)
    contact_person: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    lead_time_days: Optional[int] = Field(None, gt=0)
    reliability_rating: Optional[float] = Field(None, ge=0, le=5)

class SupplierUpdate(SupplierBase):
    pass

class SupplierRead(SupplierBase):
    id: int
    class Config:
        from_attributes = True

# --- Consumption Schemas (UPDATED: project_id optional) ---
class ConsumptionBase(BaseModel):
    material_id: int
    # Make project_id optional so clients can omit it and server can infer from material
    project_id: Optional[int] = None
    quantity_used: float = Field(..., gt=0)
    notes: Optional[str] = None

class ConsumptionCreate(ConsumptionBase):
    # date_used provided optionally
    date_used: Optional[datetime] = Field(default_factory=datetime.utcnow)

class ConsumptionRead(ConsumptionBase):
    id: int
    date_used: datetime
    material_name: Optional[str] = None
    project_name: Optional[str] = None

    class Config:
        from_attributes = True

# --- Cost Schemas ---
class CostBase(BaseModel):
    material_id: int
    supplier_id: Optional[int] = None
    unit_price: float = Field(..., ge=0)
    quantity_purchased: float = Field(..., gt=0)
    total_cost: Optional[float] = Field(None, ge=0)
    date_recorded: Optional[datetime] = None
    notes: Optional[str] = None

class CostCreate(CostBase):
    pass

class CostRead(CostBase):
    id: int
    date_recorded: datetime
    material_name: Optional[str] = None
    supplier_name: Optional[str] = None

    class Config:
        from_attributes = True

# --- Waste Schemas (Updated) ---
class WasteBase(BaseModel):
    material_id: int
    project_id: int
    quantity_wasted: float = Field(..., gt=0)
    reason: str = Field(..., min_length=1)
    preventive_measures: Optional[str] = None

class WasteCreate(WasteBase):
    pass

class WasteRead(WasteBase):
    id: int
    date_recorded: datetime
    material_name: Optional[str] = None
    project_name: Optional[str] = None

    class Config:
        from_attributes = True

# --- Alert Schemas ---
class AlertBase(BaseModel):
    material_id: int
    alert_type: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    is_active: bool = True

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
    name: Optional[str] = Field(None, min_length=1, description="New name for the project")

class ProjectRead(ProjectBase):
    id: int
    class Config:
        from_attributes = True


class SupplierScore(BaseModel):
    supplier_id: int
    supplier_name: str
    lead_time_days: Optional[int]
    reliability_rating: Optional[float]
    avg_unit_price: Optional[float]
    score: float

class ProcurementSuggestionItem(BaseModel):
    material_id: int
    material_name: str
    current_quantity: float
    days_of_stock: Optional[float]
    recommended_order_qty: Optional[float]
    reorder_point: Optional[float]
    safety_stock: Optional[float]
    reason: Optional[str]  # short rationale
    supplier_scores: Optional[List[SupplierScore]] = None
    ai_explanation: Optional[str] = None

class ProcurementSuggestionResponse(BaseModel):
    project_id: int
    project_name: Optional[str]
    generated_at: datetime
    suggestions: List[ProcurementSuggestionItem]