# main.py

import logging
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional # Import Optional
from datetime import datetime

# --- Database, Models, Schemas, CRUD ---
from database import get_db, engine
import models
# Import specific schemas needed, including Project schemas
from schemas import (
    InventoryCreate, InventoryRead, # Use InventoryRead
    SupplierCreate, SupplierRead, SupplierUpdate,
    ConsumptionCreate, ConsumptionRead,
    CostCreate, CostRead,
    WasteCreate, WasteRead,
    AlertCreate, AlertRead,
    ProjectCreate, ProjectUpdate, ProjectRead
)
# Import specific CRUD functions needed, including Project functions
from crud import (
    get_inventory, create_inventory, update_inventory_item_details, get_inventory_item, delete_inventory_item, # Use update_inventory_item_details
    get_suppliers, create_supplier_db, get_supplier_by_id, update_supplier_db, delete_supplier_db,
    get_consumption_data, create_consumption_record,
    get_cost_data, create_cost_record,
    get_waste_data, create_waste_record,
    get_alerts, create_alert, resolve_alert,
    create_project, get_projects, get_project_by_id, update_project, delete_project
)

# --- Import Agent Functions (Original Approach) ---
from agents.debris_agent import run_debris_analysis_agent
from agents.inventory_agent import run_inventory_analysis_agent

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO) # Enable basic logging
logger = logging.getLogger(__name__)

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="BuildOptima - Construction Material Manager")

# Mount static files & Setup templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.globals['now'] = datetime.utcnow

# === Page Routes (Updated) ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    """Serves the home/dashboard page."""
    try:
        # Fetch all inventory across projects for overview, consider filtering later if needed
        inventory = get_inventory(db)
        alerts = get_alerts(db)
        projects = get_projects(db, limit=5) # Get recent projects for dashboard?
    except Exception as e:
        logger.error(f"Error fetching data for home page: {e}", exc_info=True)
        inventory = []; alerts = []; projects = []
    return templates.TemplateResponse( "index.html", {"request": request, "inventory": inventory, "alerts": alerts, "projects": projects} )

@app.get("/suppliers", response_class=HTMLResponse)
async def suppliers_page(request: Request, db: Session = Depends(get_db)):
    """Serves the suppliers management page."""
    try:
        suppliers = get_suppliers(db)
    except Exception as e:
        logger.error(f"Error fetching suppliers for page: {e}", exc_info=True)
        suppliers = []
    return templates.TemplateResponse( "suppliers.html", {"request": request, "suppliers": suppliers} )

@app.get("/consumption", response_class=HTMLResponse)
async def consumption_page(request: Request, db: Session = Depends(get_db)):
    """Serves the consumption logging page, passing projects and materials list."""
    try:
        # Fetch all consumption for now, could filter by a default project later
        consumption_data = get_consumption_data(db)
        # Fetch all inventory items to populate material dropdown (needs filtering by project in JS later)
        materials = get_inventory(db)
        projects = get_projects(db) # Fetch projects for dropdown
    except Exception as e:
        logger.error(f"Error fetching data for consumption page: {e}", exc_info=True)
        consumption_data = []; materials = []; projects = []
    return templates.TemplateResponse( "consumption.html", {
        "request": request,
        "consumption_data": consumption_data,
        "materials": materials, # Pass all materials for now
        "projects": projects
    })

@app.get("/costs", response_class=HTMLResponse)
async def costs_page(request: Request, db: Session = Depends(get_db)):
    """Serves the cost tracking page."""
    try:
        cost_data = get_cost_data(db)
        materials = get_inventory(db) # Needs filtering by project in JS later if cost is project specific
        suppliers = get_suppliers(db)
    except Exception as e:
        logger.error(f"Error fetching data for costs page: {e}", exc_info=True)
        cost_data = []; materials = []; suppliers = []
    return templates.TemplateResponse( "costs.html", {"request": request, "cost_data": cost_data, "materials": materials, "suppliers": suppliers} )

@app.get("/materials", response_class=HTMLResponse)
async def materials_page(request: Request, db: Session = Depends(get_db)):
    """Serves the materials/inventory management page."""
    try:
        # Fetch all materials for now, could add project filter later
        materials = get_inventory(db)
        suppliers = get_suppliers(db) # Needed for supplier dropdown
        projects = get_projects(db) # Needed for project dropdown in add/edit
    except Exception as e:
        logger.error(f"Error fetching data for materials page: {e}", exc_info=True)
        materials = []; suppliers = []; projects = []
    return templates.TemplateResponse( "materials.html", {
        "request": request,
        "materials": materials,
        "suppliers": suppliers,
        "projects": projects # Pass projects
        })

@app.get("/waste", response_class=HTMLResponse)
async def waste_page(request: Request, db: Session = Depends(get_db)):
    """Serves the waste/debris logging page, passing projects and materials list."""
    try:
        # Fetch all waste for now, could filter by a default project later
        waste_records = get_waste_data(db)
        # Fetch all inventory items to populate material dropdown (needs filtering by project in JS later)
        materials = get_inventory(db)
        projects = get_projects(db) # Fetch projects for dropdown
    except Exception as e:
        logger.error(f"Error fetching data for waste page: {e}", exc_info=True)
        waste_records = []; materials = []; projects = []
    return templates.TemplateResponse( "waste.html", {
        "request": request,
        "waste_records": waste_records,
        "materials": materials, # Pass all materials for now
        "projects": projects
    })

@app.get("/projects", response_class=HTMLResponse)
async def projects_page(request: Request, db: Session = Depends(get_db)):
    """Serves the projects management page."""
    try:
        projects = get_projects(db)
    except Exception as e:
        logger.error(f"Error fetching projects for page: {e}", exc_info=True)
        projects = []
    return templates.TemplateResponse("projects.html", {"request": request, "projects": projects})


# === Standard CRUD API Endpoints (Updated) ===

# --- Materials ---
@app.post("/api/materials", response_model=InventoryRead, status_code=status.HTTP_201_CREATED, tags=["Materials"])
async def api_create_material(material: InventoryCreate, db: Session = Depends(get_db)):
    """Creates a new material/inventory item for a specific project."""
    try:
        new_item = create_inventory(db, material)
        # Manually populate related names for the response
        read_response = InventoryRead.from_orm(new_item)
        if new_item.supplier: read_response.supplier_name = new_item.supplier.name
        if new_item.project: read_response.project_name = new_item.project.name
        return read_response
    except (ValueError, IntegrityError) as e:
        # Catch validation errors (project/supplier not found) or unique constraint errors
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating material '{material.material_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create material.")

@app.get("/api/materials", response_model=List[InventoryRead], tags=["Materials"])
async def api_get_materials(project_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Retrieves material/inventory items, optionally filtered by project_id."""
    materials = get_inventory(db, project_id=project_id)
    # Manually populate related names for the response
    response_list = []
    for item in materials:
        read_item = InventoryRead.from_orm(item)
        if item.supplier: read_item.supplier_name = item.supplier.name
        if item.project: read_item.project_name = item.project.name
        response_list.append(read_item)
    return response_list

@app.get("/api/materials/{material_id}", response_model=InventoryRead, tags=["Materials"])
async def api_get_material(material_id: int, db: Session = Depends(get_db)):
    """Retrieves a specific material/inventory item by ID."""
    material = get_inventory_item(db, material_id) # CRUD loads relationships
    if not material: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")
    # Manually populate related names for the response
    read_response = InventoryRead.from_orm(material)
    if material.supplier: read_response.supplier_name = material.supplier.name
    if material.project: read_response.project_name = material.project.name
    return read_response

@app.put("/api/materials/{material_id}", response_model=InventoryRead, tags=["Materials"])
async def api_update_material(material_id: int, material_update: InventoryCreate, db: Session = Depends(get_db)):
    """
    Updates details of a specific material/inventory item.
    Uses InventoryCreate schema for payload, consider specific InventoryUpdate schema.
    Quantity is updated separately via consumption/waste logs.
    """
    try:
        # Use the CRUD function specifically for updating details
        updated = update_inventory_item_details(db, material_id, material_update)
        if not updated: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")
        # Manually populate related names for the response
        read_response = InventoryRead.from_orm(updated)
        if updated.supplier: read_response.supplier_name = updated.supplier.name
        if updated.project: read_response.project_name = updated.project.name
        return read_response
    except (ValueError, IntegrityError) as e:
         # Catch validation errors (project/supplier not found) or unique constraint errors
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating material details {material_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update material details.")


@app.delete("/api/materials/{material_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Materials"])
async def api_delete_material(material_id: int, db: Session = Depends(get_db)):
    """Deletes a specific material/inventory item."""
    try:
        success = delete_inventory_item(db, material_id)
        if not success: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting material ID {material_id}: {e}", exc_info=True)
        # Check if it's a constraint violation (though cascade might handle it)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete material.")
    return None

# --- Suppliers ---
# ... (Supplier API endpoints remain largely the same, ensure SupplierUpdate schema is used in PUT) ...
@app.post("/api/suppliers", response_model=SupplierRead, status_code=status.HTTP_201_CREATED, tags=["Suppliers"])
async def api_create_supplier(supplier: SupplierCreate, db: Session = Depends(get_db)):
    new_supplier = create_supplier_db(db, supplier); return SupplierRead.from_orm(new_supplier)
@app.get("/api/suppliers", response_model=List[SupplierRead], tags=["Suppliers"])
async def api_get_suppliers(db: Session = Depends(get_db)):
    suppliers = get_suppliers(db); return [SupplierRead.from_orm(s) for s in suppliers]
@app.get("/api/suppliers/{supplier_id}", response_model=SupplierRead, tags=["Suppliers"])
async def api_get_supplier(supplier_id: int, db: Session = Depends(get_db)):
    supplier = get_supplier_by_id(db, supplier_id);
    if not supplier: raise HTTPException(status_code=404, detail="Supplier not found")
    return SupplierRead.from_orm(supplier)
@app.put("/api/suppliers/{supplier_id}", response_model=SupplierRead, tags=["Suppliers"])
async def api_update_supplier(supplier_id: int, supplier: SupplierUpdate, db: Session = Depends(get_db)): # Use SupplierUpdate
    updated = update_supplier_db(db, supplier_id, supplier);
    if not updated: raise HTTPException(status_code=404, detail="Supplier not found")
    return SupplierRead.from_orm(updated)
@app.delete("/api/suppliers/{supplier_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Suppliers"])
async def api_delete_supplier(supplier_id: int, db: Session = Depends(get_db)):
    try: success = delete_supplier_db(db, supplier_id);
    except ValueError as e: raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) # Catch FK constraint error
    if not success: raise HTTPException(status_code=404, detail="Supplier not found")
    return None

# --- Consumption (Updated) ---
@app.post("/api/consumption", response_model=ConsumptionRead, status_code=status.HTTP_201_CREATED, tags=["Consumption"])
async def api_create_consumption(consumption: ConsumptionCreate, db: Session = Depends(get_db)):
    """Logs a new material consumption record, linked to a project."""
    try:
        new_record = create_consumption_record(db, consumption)
        # Manually populate related names for the response
        read_response = ConsumptionRead.from_orm(new_record)
        if new_record.material: read_response.material_name = new_record.material.material_name
        if new_record.project_rel: read_response.project_name = new_record.project_rel.name
        return read_response
    except ValueError as e: # Catch errors like insufficient inventory or project/material not found
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # db.rollback() # Rollback might happen in CRUD, confirm
        logger.error(f"Error creating consumption record: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to log consumption.")

@app.get("/api/consumption", response_model=List[ConsumptionRead], tags=["Consumption"])
async def api_get_consumption(project_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Retrieves consumption records, optionally filtered by project_id."""
    consumption_data = get_consumption_data(db, project_id=project_id) # Pass filter
    response_list = []
    for item in consumption_data:
        read_item = ConsumptionRead.from_orm(item)
        # Populate related names from eager loaded relationships
        if item.material: read_item.material_name = item.material.material_name
        if item.project_rel: read_item.project_name = item.project_rel.name
        response_list.append(read_item)
    return response_list

# --- Costs ---
# (Cost endpoints remain largely the same, ensure relationships are populated for Read)
@app.post("/api/costs", response_model=CostRead, status_code=status.HTTP_201_CREATED, tags=["Costs"])
async def api_create_cost(cost: CostCreate, db: Session = Depends(get_db)):
    """Logs a new cost record."""
    try:
         new_record = create_cost_record(db, cost)
         # Manually populate related names for the response
         read_response = CostRead.from_orm(new_record)
         if new_record.material: read_response.material_name = new_record.material.material_name
         if new_record.supplier: read_response.supplier_name = new_record.supplier.name
         return read_response
    except ValueError as e: # Catch FK validation errors
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating cost record: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to log cost.")

@app.get("/api/costs", response_model=List[CostRead], tags=["Costs"])
async def api_get_costs(db: Session = Depends(get_db)):
    """Retrieves all cost records."""
    cost_data = get_cost_data(db) # CRUD function eager loads
    response_list = []
    for item in cost_data:
        read_item = CostRead.from_orm(item)
        if item.material: read_item.material_name = item.material.material_name
        if item.supplier: read_item.supplier_name = item.supplier.name
        response_list.append(read_item)
    return response_list

# --- Waste (Updated) ---
@app.post("/api/waste", response_model=WasteRead, status_code=status.HTTP_201_CREATED, tags=["Waste"])
async def api_create_waste(waste: WasteCreate, db: Session = Depends(get_db)):
    """Logs a new waste record, linked to a project."""
    try:
        new_waste_record = create_waste_record(db, waste)
        # Manually populate related names for the response
        read_response = WasteRead.from_orm(new_waste_record)
        if new_waste_record.material: read_response.material_name = new_waste_record.material.material_name
        if new_waste_record.project_rel: read_response.project_name = new_waste_record.project_rel.name
        return read_response
    except ValueError as e: # Catch material/project not found errors
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # db.rollback() # Rollback might happen in CRUD, confirm
        logger.error(f"Error creating waste record: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to log waste.")


@app.get("/api/waste", response_model=List[WasteRead], tags=["Waste"])
async def api_get_waste(project_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Retrieves waste records, optionally filtered by project_id."""
    waste_data = get_waste_data(db, project_id=project_id) # Pass filter
    response_list = []
    for item in waste_data:
        read_item = WasteRead.from_orm(item)
        # Populate related names from eager loaded relationships
        if item.material: read_item.material_name = item.material.material_name
        if item.project_rel: read_item.project_name = item.project_rel.name
        response_list.append(read_item)
    return response_list

# --- Projects ---
# (Project endpoints remain the same as previously defined)
@app.post("/api/projects", response_model=ProjectRead, status_code=status.HTTP_201_CREATED, tags=["Projects"])
async def api_create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    try: db_project = create_project(db=db, project=project); return ProjectRead.from_orm(db_project)
    except IntegrityError as e: db.rollback(); err_detail = str(e.orig) if e.orig else str(e); status_code = status.HTTP_409_CONFLICT if "UNIQUE constraint failed: projects.name" in err_detail else status.HTTP_400_BAD_REQUEST; detail = f"Project with name '{project.name}' already exists." if status_code == 409 else "Database integrity error." ; logger.warning(f"IntegrityError creating project '{project.name}': {err_detail}"); raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e: db.rollback(); logger.exception(f"Unexpected error creating project: {project.name}"); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error creating project.")
@app.get("/api/projects", response_model=List[ProjectRead], tags=["Projects"])
async def api_get_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    projects = get_projects(db=db, skip=skip, limit=limit); return [ProjectRead.from_orm(p) for p in projects]
@app.get("/api/projects/{project_id}", response_model=ProjectRead, tags=["Projects"])
async def api_get_project(project_id: int, db: Session = Depends(get_db)):
    db_project = get_project_by_id(db=db, project_id=project_id);
    if db_project is None: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return ProjectRead.from_orm(db_project)
@app.put("/api/projects/{project_id}", response_model=ProjectRead, tags=["Projects"])
async def api_update_project(project_id: int, project: ProjectUpdate, db: Session = Depends(get_db)):
    try: updated_project = update_project(db=db, project_id=project_id, project_update=project);
    except IntegrityError as e: db.rollback(); err_detail = str(e.orig) if e.orig else str(e); status_code = status.HTTP_409_CONFLICT if "UNIQUE constraint failed: projects.name" in err_detail else status.HTTP_400_BAD_REQUEST; detail="Cannot update project: Another project with the provided name might already exist." if status_code == 409 else "Database integrity error."; logger.warning(f"IntegrityError updating project ID {project_id}: {err_detail}"); raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e: db.rollback(); logger.exception(f"Error updating project ID: {project_id}"); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error updating project.")
    if updated_project is None: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return ProjectRead.from_orm(updated_project)
@app.delete("/api/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Projects"])
async def api_delete_project(project_id: int, db: Session = Depends(get_db)):
    try: success = delete_project(db=db, project_id=project_id);
    except ValueError as e: raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) # Catch FK constraint error from CRUD
    except Exception as e: db.rollback(); logger.exception(f"Error deleting project ID: {project_id}"); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error deleting project.")
    if not success: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return None


# === Agent API Endpoints (Original Approach) ===
@app.get("/api/debris/report", response_class=JSONResponse, tags=["AI Agents"])
async def get_debris_analysis_report():
    """ Triggers the debris analysis agent and returns the report. """
    logger.info("Received request for debris analysis report.")
    try: report = run_debris_analysis_agent(); logger.info("Debris analysis report generated successfully."); return JSONResponse(content={"report": report})
    except Exception as e: logger.exception("Error running debris analysis agent"); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate debris report: {str(e)}")
@app.get("/api/inventory/report", response_class=JSONResponse, tags=["AI Agents"])
async def get_inventory_analysis_report():
    """ Triggers the inventory analysis agent and returns the report. """
    logger.info("Received request for inventory analysis report.")
    try: report = run_inventory_analysis_agent(); logger.info("Inventory analysis report generated successfully."); return JSONResponse(content={"report": report})
    except Exception as e: logger.exception("Error running inventory analysis agent"); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate inventory report: {str(e)}")

