from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Inventory(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True, index=True)
    material_name = Column(String, index=True)
    quantity = Column(Float)
    unit = Column(String)
    reorder_point = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"))

    supplier = relationship("Supplier", back_populates="materials")
    consumption_records = relationship("Consumption", back_populates="material")
    waste_records = relationship("Waste", back_populates="material")
    cost_records = relationship("Cost", back_populates="material")

class Supplier(Base):
    __tablename__ = "suppliers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    contact_person = Column(String)
    email = Column(String)
    phone = Column(String)
    address = Column(String)
    lead_time_days = Column(Integer)
    reliability_rating = Column(Float)

    materials = relationship("Inventory", back_populates="supplier")
    cost_records = relationship("Cost", back_populates="supplier")

class Consumption(Base):
    __tablename__ = "consumption"

    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("inventory.id"))
    project = Column(String)  # Field renamed to match schema & form
    quantity_used = Column(Float)
    date_used = Column(DateTime, default=datetime.utcnow)
    notes = Column(String, nullable=True)

    material = relationship("Inventory", back_populates="consumption_records")

class Cost(Base):
    __tablename__ = "costs"

    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("inventory.id"))
    supplier_id = Column(Integer, ForeignKey("suppliers.id"))
    unit_price = Column(Float)
    quantity_purchased = Column(Float)
    total_cost = Column(Float)
    date_recorded = Column(DateTime, default=datetime.utcnow)
    notes = Column(String, nullable=True)  # Added notes field

    material = relationship("Inventory", back_populates="cost_records")
    supplier = relationship("Supplier", back_populates="cost_records")

class Waste(Base):
    __tablename__ = "waste"

    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("inventory.id"))
    project_name = Column(String)
    quantity_wasted = Column(Float)
    date_recorded = Column(DateTime, default=datetime.utcnow)
    reason = Column(String)
    preventive_measures = Column(String, nullable=True)

    material = relationship("Inventory", back_populates="waste_records")

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("inventory.id"))
    alert_type = Column(String)  # e.g. "low_stock"
    message = Column(String)
    date_created = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)  # 1 = active, 0 = resolved
