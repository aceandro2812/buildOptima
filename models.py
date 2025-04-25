# models.py

# Import necessary types from SQLAlchemy
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Date, Text, Boolean
)
from sqlalchemy.orm import relationship, joinedload # Import relationship and joinedload
from datetime import datetime, date
from database import Base # Import Base from your database setup

# --- Project Model ---
# Define Project first as other models will refer to it
class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True, nullable=False)
    location = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    status = Column(String, default="Planning", nullable=False) # e.g., Planning, In Progress, Completed, On Hold

    # Relationships to other tables that link TO this project
    # cascade="all, delete-orphan": If a project is deleted, its related records are also deleted.
    # Use with caution, maybe prevent project deletion if linked records exist instead.
    inventory_items = relationship("Inventory", back_populates="project", cascade="all, delete-orphan")
    consumption_records = relationship("Consumption", back_populates="project_rel", cascade="all, delete-orphan")
    waste_records = relationship("Waste", back_populates="project_rel", cascade="all, delete-orphan")


# --- Inventory Model (Updated) ---
class Inventory(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True, index=True)
    material_name = Column(String, index=True, nullable=False)
    quantity = Column(Float, nullable=False, default=0.0)
    unit = Column(String, nullable=False)
    reorder_point = Column(Float, nullable=False, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) # Add onupdate
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=True) # Supplier might be optional?

    # --- Link to Project ---
    # Assuming inventory is project-specific based on your requirement
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False) # Make non-nullable if inventory MUST belong to a project
    project = relationship("Project", back_populates="inventory_items")
    # --- End Project Link ---

    # Relationships
    supplier = relationship("Supplier", back_populates="materials")
    # If an Inventory item is deleted, delete its related records
    consumption_records = relationship("Consumption", back_populates="material", cascade="all, delete-orphan")
    waste_records = relationship("Waste", back_populates="material", cascade="all, delete-orphan")
    cost_records = relationship("Cost", back_populates="material", cascade="all, delete-orphan")


# --- Supplier Model ---
class Supplier(Base):
    __tablename__ = "suppliers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    contact_person = Column(String, nullable=True) # Allow nullable
    email = Column(String, nullable=True) # Allow nullable
    phone = Column(String, nullable=True) # Allow nullable
    address = Column(String, nullable=True) # Allow nullable
    lead_time_days = Column(Integer, nullable=True) # Allow nullable
    reliability_rating = Column(Float, nullable=True) # Allow nullable

    # Relationships
    materials = relationship("Inventory", back_populates="supplier") # Default RESTRICT on delete
    cost_records = relationship("Cost", back_populates="supplier") # Default RESTRICT on delete


# --- Consumption Model (Updated) ---
class Consumption(Base):
    __tablename__ = "consumption"
    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("inventory.id"), nullable=False) # Must link to a material
    quantity_used = Column(Float, nullable=False)
    date_used = Column(DateTime, default=datetime.utcnow, nullable=False)
    notes = Column(String, nullable=True)

    # --- Link to Project ---
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False) # Must link to a project
    project_rel = relationship("Project", back_populates="consumption_records")
    # Removed old 'project' string field

    # Relationships
    material = relationship("Inventory", back_populates="consumption_records")


# --- Cost Model ---
class Cost(Base):
    __tablename__ = "costs"
    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("inventory.id"), nullable=False) # Must link to a material
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=True) # Cost might not always have a supplier?
    unit_price = Column(Float, nullable=False)
    quantity_purchased = Column(Float, nullable=False)
    total_cost = Column(Float, nullable=False) # Calculated, should be stored
    date_recorded = Column(DateTime, default=datetime.utcnow, nullable=False)
    notes = Column(String, nullable=True)

    # Relationships
    material = relationship("Inventory", back_populates="cost_records")
    supplier = relationship("Supplier", back_populates="cost_records")


# --- Waste Model (Updated) ---
class Waste(Base):
    __tablename__ = "waste"
    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("inventory.id"), nullable=False) # Must link to a material
    quantity_wasted = Column(Float, nullable=False)
    date_recorded = Column(DateTime, default=datetime.utcnow, nullable=False)
    reason = Column(String, nullable=False)
    preventive_measures = Column(String, nullable=True)

    # --- Link to Project ---
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False) # Must link to a project
    project_rel = relationship("Project", back_populates="waste_records")
    # Removed old 'project_name' string field

    # Relationships
    material = relationship("Inventory", back_populates="waste_records")


# --- Alert Model ---
class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("inventory.id"), nullable=False) # Must link to a material
    alert_type = Column(String, nullable=False)
    message = Column(String, nullable=False)
    date_created = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False) # Use Boolean

    # Relationship (Optional: If you want to access material from alert)
    material = relationship("Inventory")

