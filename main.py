# main.py

import logging
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import asyncio
from typing import List, Optional
from datetime import datetime
from broadcast import broadcast_manager
from fastapi import WebSocket, WebSocketDisconnect


# --- Database, Models, Schemas, CRUD ---
from database import get_db, engine
import models
from schemas import (
    InventoryCreate, InventoryRead,
    SupplierCreate, SupplierRead, SupplierUpdate,
    ConsumptionCreate, ConsumptionRead,
    CostCreate, CostRead,
    WasteCreate, WasteRead,
    ProjectCreate, ProjectUpdate, ProjectRead
)
from crud import (
    get_inventory, create_inventory, update_inventory_item_details, get_inventory_item, delete_inventory_item,
    get_suppliers, create_supplier_db, get_supplier_by_id, update_supplier_db, delete_supplier_db,
    get_consumption_data, create_consumption_record,
    get_cost_data, create_cost_record,
    get_waste_data, create_waste_record,
    get_alerts, create_project, get_projects, get_project_by_id, update_project, delete_project, get_resource_limits_report,
    crud_get_total_materials, crud_get_active_alerts_count, crud_get_num_exceeded, crud_get_total_cost, crud_get_top_overruns, crud_get_recent_consumption
)

# --- Import Agent Functions ---
from agents.debris_agent import run_debris_analysis_agent
from agents.inventory_agent import run_inventory_analysis_agent
from agents.procurement_agent import run_procurement_advisor

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
models.Base.metadata.create_all(bind=engine)

# --- Serialization Helper Functions ---
def serialize_cost(cost):
    """Convert Cost SQLAlchemy object to dictionary for JSON serialization."""
    # Handle date serialization safely
    date_recorded = None
    if cost.date_recorded:
        if hasattr(cost.date_recorded, 'isoformat'):
            date_recorded = cost.date_recorded.isoformat()
        else:
            date_recorded = str(cost.date_recorded)
    
    return {
        'id': cost.id,
        'material_id': cost.material_id,
        'supplier_id': cost.supplier_id,
        'unit_price': float(cost.unit_price) if cost.unit_price else 0.0,
        'quantity_purchased': float(cost.quantity_purchased) if cost.quantity_purchased else 0.0,
        'total_cost': float(cost.total_cost) if cost.total_cost else 0.0,
        'date_recorded': date_recorded,
        'notes': cost.notes,
        'material': {
            'material_name': cost.material.material_name if cost.material else None,
            'unit': cost.material.unit if cost.material else None
        } if cost.material else None,
        'supplier': {
            'name': cost.supplier.name if cost.supplier else None
        } if cost.supplier else None
    }

def serialize_consumption(consumption):
    """Convert Consumption SQLAlchemy object to dictionary for JSON serialization."""
    # Handle date serialization safely
    date_used = None
    if consumption.date_used:
        if hasattr(consumption.date_used, 'isoformat'):
            date_used = consumption.date_used.isoformat()
        else:
            date_used = str(consumption.date_used)
    
    return {
        'id': consumption.id,
        'material_id': consumption.material_id,
        'project_id': consumption.project_id,
        'quantity_used': float(consumption.quantity_used) if consumption.quantity_used else 0.0,
        'date_used': date_used,
        'notes': consumption.notes,
        'material': {
            'material_name': consumption.material.material_name if consumption.material else None,
            'unit': consumption.material.unit if consumption.material else None
        } if consumption.material else None,
        'project_rel': {
            'name': consumption.project_rel.name if consumption.project_rel else None
        } if consumption.project_rel else None
    }

def serialize_waste(waste):
    """Convert Waste SQLAlchemy object to dictionary for JSON serialization."""
    # Handle date serialization safely
    date_wasted = None
    if waste.date_recorded:
        if hasattr(waste.date_recorded, 'isoformat'):
            date_wasted = waste.date_recorded.isoformat()
        else:
            date_wasted = str(waste.date_recorded)
    
    return {
        'id': waste.id,
        'material_id': waste.material_id,
        'project_id': waste.project_id,
        'quantity_wasted': float(waste.quantity_wasted) if waste.quantity_wasted else 0.0,
        'date_wasted': date_wasted,
        'reason': waste.reason,
        'preventive_measures': waste.preventive_measures,
        'material_name': waste.material.material_name if waste.material else None,
        'project_name': waste.project_rel.name if waste.project_rel else None,
        'material': {
            'material_name': waste.material.material_name if waste.material else None
        } if waste.material else None,
        'project_rel': {
            'name': waste.project_rel.name if waste.project_rel else None
        } if waste.project_rel else None
    }

app = FastAPI(title="BuildOptima - Construction Material Manager")

# Mount static files & Setup templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.globals['now'] = datetime.utcnow

# === Page Routes ===
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request, db: Session = Depends(get_db)):
    projects = get_projects(db)
    return templates.TemplateResponse("dashboard.html", {"request": request, "projects": projects})

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """
    WebSocket for dashboard real-time updates.
    Clients should connect with: new WebSocket(`ws://<host>/ws/dashboard`)
    Server will send JSON messages { type: "<event_type>", payload: { ... } }.
    """
    await broadcast_manager.connect(websocket)
    try:
        # Keep connection alive; client won't send on this socket normally.
        while True:
            # receive_text to keep the socket alive and detect client pings
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                # ignore other client-side noises; just continue
                await asyncio.sleep(0.1)
    finally:
        await broadcast_manager.disconnect(websocket)

@app.get("/api/dashboard/snapshot")
def api_dashboard_snapshot(project_id: int = None, db: Session = Depends(get_db)):
    """
    Returns aggregated metrics needed to initialize the dashboard.
    If project_id is provided, data is scoped to that project.
    """
    # use helper functions in crud.py (we'll add these)
    try:
        data = {
            "summary": {
                "total_materials": crud_get_total_materials(db, project_id),
                "active_alerts": crud_get_active_alerts_count(db, project_id),
                "num_exceeded": crud_get_num_exceeded(db, project_id),
                "total_cost": crud_get_total_cost(db, project_id)
            },
            "top_overruns": crud_get_top_overruns(db, project_id, limit=5),
            "recent_consumption": crud_get_recent_consumption(db, project_id, limit=10),
            "resource_limits": get_resource_limits_report(db, project_id) if project_id else None
        }
        return JSONResponse(content=data)
    except Exception:
        logger.exception("Failed to get dashboard snapshot")
        raise HTTPException(status_code=500, detail="Failed to get dashboard snapshot")
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    """Serves the home/dashboard page."""
    try:
        inventory = get_inventory(db)
        alerts = get_alerts(db)
        projects = get_projects(db, limit=5)
    except Exception as e:
        logger.error(f"Error fetching data for home page: {e}", exc_info=True)
        inventory = []
        alerts = []
        projects = []
    return templates.TemplateResponse("index.html", {"request": request, "inventory": inventory, "alerts": alerts, "projects": projects})

@app.get("/suppliers", response_class=HTMLResponse)
async def suppliers_page(request: Request, db: Session = Depends(get_db)):
    """Serves the suppliers management page."""
    try:
        suppliers = get_suppliers(db)
    except Exception as e:
        logger.error(f"Error fetching suppliers for page: {e}", exc_info=True)
        suppliers = []
    return templates.TemplateResponse("suppliers.html", {"request": request, "suppliers": suppliers})

@app.get("/consumption", response_class=HTMLResponse)
async def consumption_page(request: Request, db: Session = Depends(get_db)):
    """Serves the consumption logging page, passing projects and materials list."""
    try:
        consumption_data_raw = get_consumption_data(db)
        consumption_data = [serialize_consumption(c) for c in consumption_data_raw]
        materials = get_inventory(db)
        projects = get_projects(db)
    except Exception as e:
        logger.error(f"Error fetching data for consumption page: {e}", exc_info=True)
        consumption_data = []
        materials = []
        projects = []
    return templates.TemplateResponse("consumption.html", {
        "request": request,
        "consumption_data": consumption_data,
        "materials": materials,
        "projects": projects
    })

@app.get("/costs", response_class=HTMLResponse)
async def costs_page(request: Request, db: Session = Depends(get_db)):
    """Serves the cost tracking page."""
    try:
        cost_data_raw = get_cost_data(db)
        cost_data = [serialize_cost(c) for c in cost_data_raw]
        materials = get_inventory(db)
        suppliers = get_suppliers(db)
    except Exception as e:
        logger.error(f"Error fetching data for costs page: {e}", exc_info=True)
        cost_data = []
        materials = []
        suppliers = []
    return templates.TemplateResponse("costs.html", {"request": request, "cost_data": cost_data, "materials": materials, "suppliers": suppliers})

@app.get("/materials", response_class=HTMLResponse)
async def materials_page(request: Request, db: Session = Depends(get_db)):
    """Serves the materials/inventory management page."""
    try:
        materials = get_inventory(db)
        suppliers = get_suppliers(db)
        projects = get_projects(db)
    except Exception as e:
        logger.error(f"Error fetching data for materials page: {e}", exc_info=True)
        materials = []
        suppliers = []
        projects = []
    return templates.TemplateResponse("materials.html", {
        "request": request,
        "materials": materials,
        "suppliers": suppliers,
        "projects": projects
    })

@app.get("/waste", response_class=HTMLResponse)
async def waste_page(request: Request, db: Session = Depends(get_db)):
    """Serves the waste/debris logging page, passing projects and materials list."""
    try:
        waste_records_raw = get_waste_data(db)
        waste_records = [serialize_waste(w) for w in waste_records_raw]
        materials = get_inventory(db)
        projects = get_projects(db)
    except Exception as e:
        logger.error(f"Error fetching data for waste page: {e}", exc_info=True)
        waste_records = []
        materials = []
        projects = []
    return templates.TemplateResponse("waste.html", {
        "request": request,
        "waste_records": waste_records,
        "materials": materials,
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

# === Standard CRUD API Endpoints ===

# --- Materials ---
@app.post("/api/materials", response_model=InventoryRead, status_code=status.HTTP_201_CREATED, tags=["Materials"])
async def api_create_material(material: InventoryCreate, db: Session = Depends(get_db)):
    """Creates a new material/inventory item for a specific project."""
    try:
        new_item = create_inventory(db, material)
        read_response = InventoryRead.from_orm(new_item)
        if new_item.supplier:
            read_response.supplier_name = new_item.supplier.name
        if new_item.project:
            read_response.project_name = new_item.project.name
        return read_response
    except (ValueError, IntegrityError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating material '{material.material_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create material.")

@app.get("/api/materials", response_model=List[InventoryRead], tags=["Materials"])
async def api_get_materials(project_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Retrieves material/inventory items, optionally filtered by project_id."""
    materials = get_inventory(db, project_id=project_id)
    response_list = []
    for item in materials:
        read_item = InventoryRead.from_orm(item)
        if item.supplier:
            read_item.supplier_name = item.supplier.name
        if item.project:
            read_item.project_name = item.project.name
        response_list.append(read_item)
    return response_list

@app.get("/api/materials/{material_id}", response_model=InventoryRead, tags=["Materials"])
async def api_get_material(material_id: int, db: Session = Depends(get_db)):
    """Retrieves a specific material/inventory item by ID."""
    material = get_inventory_item(db, material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")
    read_response = InventoryRead.from_orm(material)
    if material.supplier:
        read_response.supplier_name = material.supplier.name
    if material.project:
        read_response.project_name = material.project.name
    return read_response

@app.put("/api/materials/{material_id}", response_model=InventoryRead, tags=["Materials"])
async def api_update_material(material_id: int, material_update: InventoryCreate, db: Session = Depends(get_db)):
    """Updates details of a specific material/inventory item."""
    try:
        updated = update_inventory_item_details(db, material_id, material_update)
        if not updated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")
        read_response = InventoryRead.from_orm(updated)
        if updated.supplier:
            read_response.supplier_name = updated.supplier.name
        if updated.project:
            read_response.project_name = updated.project.name
        return read_response
    except (ValueError, IntegrityError) as e:
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
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting material ID {material_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete material.")
    return None

# --- Suppliers ---
@app.post("/api/suppliers", response_model=SupplierRead, status_code=status.HTTP_201_CREATED, tags=["Suppliers"])
async def api_create_supplier(supplier: SupplierCreate, db: Session = Depends(get_db)):
    new_supplier = create_supplier_db(db, supplier)
    return SupplierRead.from_orm(new_supplier)

@app.get("/api/suppliers", response_model=List[SupplierRead], tags=["Suppliers"])
async def api_get_suppliers(db: Session = Depends(get_db)):
    suppliers = get_suppliers(db)
    return [SupplierRead.from_orm(s) for s in suppliers]

@app.get("/api/suppliers/{supplier_id}", response_model=SupplierRead, tags=["Suppliers"])
async def api_get_supplier(supplier_id: int, db: Session = Depends(get_db)):
    supplier = get_supplier_by_id(db, supplier_id)
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return SupplierRead.from_orm(supplier)

@app.put("/api/suppliers/{supplier_id}", response_model=SupplierRead, tags=["Suppliers"])
async def api_update_supplier(supplier_id: int, supplier: SupplierUpdate, db: Session = Depends(get_db)):
    updated = update_supplier_db(db, supplier_id, supplier)
    if not updated:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return SupplierRead.from_orm(updated)

@app.delete("/api/suppliers/{supplier_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Suppliers"])
async def api_delete_supplier(supplier_id: int, db: Session = Depends(get_db)):
    try:
        success = delete_supplier_db(db, supplier_id)
        if not success:
            raise HTTPException(status_code=404, detail="Supplier not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    return None

# --- Consumption ---
@app.post("/api/consumption", response_model=ConsumptionRead, status_code=status.HTTP_201_CREATED, tags=["Consumption"])
async def api_create_consumption(consumption: ConsumptionCreate, db: Session = Depends(get_db)):
    """Logs a new material consumption record, linked to a project."""
    try:
        new_record = create_consumption_record(db, consumption)
        read_response = ConsumptionRead.from_orm(new_record)
        if new_record.material:
            read_response.material_name = new_record.material.material_name
        if new_record.project_rel:
            read_response.project_name = new_record.project_rel.name
        return read_response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating consumption record: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to log consumption.")

@app.get("/api/consumption", response_model=List[ConsumptionRead], tags=["Consumption"])
async def api_get_consumption(project_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Retrieves consumption records, optionally filtered by project_id."""
    consumption_data = get_consumption_data(db, project_id=project_id)
    response_list = []
    for item in consumption_data:
        read_item = ConsumptionRead.from_orm(item)
        if item.material:
            read_item.material_name = item.material.material_name
        if item.project_rel:
            read_item.project_name = item.project_rel.name
        response_list.append(read_item)
    return response_list

# --- Costs ---
@app.post("/api/costs", response_model=CostRead, status_code=status.HTTP_201_CREATED, tags=["Costs"])
async def api_create_cost(cost: CostCreate, db: Session = Depends(get_db)):
    """Logs a new cost record."""
    try:
        new_record = create_cost_record(db, cost)
        read_response = CostRead.from_orm(new_record)
        if new_record.material:
            read_response.material_name = new_record.material.material_name
        if new_record.supplier:
            read_response.supplier_name = new_record.supplier.name
        return read_response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating cost record: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to log cost.")

@app.get("/api/costs", response_model=List[CostRead], tags=["Costs"])
async def api_get_costs(db: Session = Depends(get_db)):
    """Retrieves all cost records."""
    cost_data = get_cost_data(db)
    response_list = []
    for item in cost_data:
        read_item = CostRead.from_orm(item)
        if item.material:
            read_item.material_name = item.material.material_name
        if item.supplier:
            read_item.supplier_name = item.supplier.name
        response_list.append(read_item)
    return response_list

# --- Waste ---
@app.post("/api/waste", response_model=WasteRead, status_code=status.HTTP_201_CREATED, tags=["Waste"])
async def api_create_waste(waste: WasteCreate, db: Session = Depends(get_db)):
    """Logs a new waste record, linked to a project."""
    try:
        new_waste_record = create_waste_record(db, waste)
        read_response = WasteRead.from_orm(new_waste_record)
        if new_waste_record.material:
            read_response.material_name = new_waste_record.material.material_name
        if new_waste_record.project_rel:
            read_response.project_name = new_waste_record.project_rel.name
        return read_response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating waste record: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to log waste.")

@app.get("/api/waste", response_model=List[WasteRead], tags=["Waste"])
async def api_get_waste(project_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Retrieves waste records, optionally filtered by project_id."""
    waste_data = get_waste_data(db, project_id=project_id)
    response_list = []
    for item in waste_data:
        read_item = WasteRead.from_orm(item)
        if item.material:
            read_item.material_name = item.material.material_name
        if item.project_rel:
            read_item.project_name = item.project_rel.name
        response_list.append(read_item)
    return response_list

# --- Projects ---
@app.post("/api/projects", response_model=ProjectRead, status_code=status.HTTP_201_CREATED, tags=["Projects"])
async def api_create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    try:
        db_project = create_project(db=db, project=project)
        return ProjectRead.from_orm(db_project)
    except IntegrityError as e:
        db.rollback()
        err_detail = str(e.orig) if e.orig else str(e)
        status_code = status.HTTP_409_CONFLICT if "UNIQUE constraint failed: projects.name" in err_detail else status.HTTP_400_BAD_REQUEST
        detail = f"Project with name '{project.name}' already exists." if status_code == 409 else "Database integrity error."
        logger.warning(f"IntegrityError creating project '{project.name}': {err_detail}")
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception:
        db.rollback()
        logger.exception(f"Unexpected error creating project: {project.name}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error creating project.")

@app.get("/api/projects", response_model=List[ProjectRead], tags=["Projects"])
async def api_get_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    projects = get_projects(db=db, skip=skip, limit=limit)
    return [ProjectRead.from_orm(p) for p in projects]

@app.get("/api/projects/{project_id}", response_model=ProjectRead, tags=["Projects"])
async def api_get_project(project_id: int, db: Session = Depends(get_db)):
    db_project = get_project_by_id(db=db, project_id=project_id)
    if db_project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return ProjectRead.from_orm(db_project)

@app.put("/api/projects/{project_id}", response_model=ProjectRead, tags=["Projects"])
async def api_update_project(project_id: int, project: ProjectUpdate, db: Session = Depends(get_db)):
    try:
        updated_project = update_project(db=db, project_id=project_id, project_update=project)
    except IntegrityError as e:
        db.rollback()
        err_detail = str(e.orig) if e.orig else str(e)
        status_code = status.HTTP_409_CONFLICT if "UNIQUE constraint failed: projects.name" in err_detail else status.HTTP_400_BAD_REQUEST
        detail="Cannot update project: Another project with the provided name might already exist." if status_code == 409 else "Database integrity error."
        logger.warning(f"IntegrityError updating project ID {project_id}: {err_detail}")
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception:
        db.rollback()
        logger.exception(f"Error updating project ID: {project_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error updating project.")
    if updated_project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return ProjectRead.from_orm(updated_project)

@app.delete("/api/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Projects"])
async def api_delete_project(project_id: int, db: Session = Depends(get_db)):
    try:
        success = delete_project(db=db, project_id=project_id)
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception:
        db.rollback()
        logger.exception(f"Error deleting project ID: {project_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error deleting project.")
    return None

# === GIS API Endpoints ===
@app.get("/api/gis/project-locations", response_class=JSONResponse, tags=["GIS"])
async def api_get_project_locations(db: Session = Depends(get_db)):
    """Get all projects with their location coordinates for map display."""
    try:
        projects = get_projects(db=db)
        
        # Return simplified project data with coordinates
        locations = []
        for project in projects:
            project_data = {
                "id": project.id,
                "name": project.name,
                "status": project.status,
                "location": project.location,  # Text location
                "latitude": project.latitude,
                "longitude": project.longitude
            }
            locations.append(project_data)
        
        return locations
        
    except Exception as e:
        logger.exception("Error fetching project locations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to fetch project locations"
        )

@app.get("/api/gis/supplier-locations", response_class=JSONResponse, tags=["GIS"])
async def api_get_supplier_locations(db: Session = Depends(get_db)):
    """Get all suppliers with their location coordinates for map display."""
    try:
        suppliers = get_suppliers(db=db)
        
        # Return simplified supplier data with coordinates
        locations = []
        for supplier in suppliers:
            supplier_data = {
                "id": supplier.id,
                "name": supplier.name,
                "contact_person": supplier.contact_person,
                "phone": supplier.phone,
                "address": supplier.address,  # Text address
                "latitude": supplier.latitude,
                "longitude": supplier.longitude
            }
            locations.append(supplier_data)
        
        return locations
        
    except Exception as e:
        logger.exception("Error fetching supplier locations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to fetch supplier locations"
        )

@app.get("/api/gis/project-distances/{project_id}", response_class=JSONResponse, tags=["GIS"])
async def api_get_project_distances(project_id: int, db: Session = Depends(get_db)):
    """Get distances from a project to all suppliers."""
    try:
        # Import distance calculator
        from distance_utils import calculate_project_supplier_distances
        
        # Get project
        project = get_project_by_id(db=db, project_id=project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.latitude or not project.longitude:
            raise HTTPException(status_code=400, detail="Project has no coordinates")
        
        # Get all suppliers
        suppliers = get_suppliers(db=db)
        
        # Calculate distances
        suppliers_with_distance = calculate_project_supplier_distances(
            project.latitude, project.longitude, suppliers
        )
        
        return {
            "project": {
                "id": project.id,
                "name": project.name,
                "latitude": project.latitude,
                "longitude": project.longitude
            },
            "suppliers": suppliers_with_distance
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error calculating project distances")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to calculate distances"
        )

@app.get("/api/gis/nearest-suppliers/{project_id}", response_class=JSONResponse, tags=["GIS"])
async def api_get_nearest_suppliers(project_id: int, limit: int = 5, db: Session = Depends(get_db)):
    """Get nearest suppliers to a project."""
    try:
        # Import distance calculator
        from distance_utils import find_nearest_suppliers
        
        # Get project
        project = get_project_by_id(db=db, project_id=project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.latitude or not project.longitude:
            raise HTTPException(status_code=400, detail="Project has no coordinates")
        
        # Get all suppliers
        suppliers = get_suppliers(db=db)
        
        # Find nearest suppliers
        nearest_suppliers = find_nearest_suppliers(
            project.latitude, project.longitude, suppliers, limit
        )
        
        return {
            "project": {
                "id": project.id,
                "name": project.name,
                "latitude": project.latitude,
                "longitude": project.longitude
            },
            "nearest_suppliers": nearest_suppliers,
            "total_suppliers_with_coordinates": len([s for s in suppliers if s.latitude and s.longitude])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error finding nearest suppliers")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to find nearest suppliers"
        )

@app.get("/api/gis/distance-matrix", response_class=JSONResponse, tags=["GIS"])
async def api_get_distance_matrix(db: Session = Depends(get_db)):
    """Get distance matrix between all projects and suppliers."""
    try:
        # Import distance calculator
        from distance_utils import calculate_project_supplier_distances
        
        def calculate_distance_matrix(projects, suppliers):
            distance_matrix = {}
            for project in projects:
                if project.latitude is not None and project.longitude is not None:
                    suppliers_with_distance = calculate_project_supplier_distances(
                        project.latitude, project.longitude, suppliers
                    )
                    distance_matrix[project.id] = suppliers_with_distance
            return distance_matrix
        
        # Get all projects and suppliers
        projects = get_projects(db=db)
        suppliers = get_suppliers(db=db)
        
        # Calculate distance matrix
        distance_matrix = calculate_distance_matrix(projects, suppliers)
        
        return {
            "projects_count": len([p for p in projects if p.latitude and p.longitude]),
            "suppliers_count": len([s for s in suppliers if s.latitude and s.longitude]),
            "distance_matrix": distance_matrix
        }
        
    except Exception as e:
        logger.exception("Error calculating distance matrix")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to calculate distance matrix"
        )

# === Alerts API Endpoints ===
@app.get("/api/alerts", response_class=JSONResponse, tags=["Alerts"])
async def api_get_alerts(project_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Get active alerts, optionally filtered by project."""
    try:
        alerts = get_alerts(db)
        
        # Filter by project if specified
        if project_id:
            # Filter alerts for materials in the specified project
            filtered_alerts = []
            for alert in alerts:
                if alert.material and alert.material.project_id == project_id:
                    filtered_alerts.append(alert)
            alerts = filtered_alerts
        
        # Convert to JSON-serializable format
        alerts_data = []
        for alert in alerts:
            alert_data = {
                "id": alert.id,
                "alert_type": alert.alert_type,
                "message": alert.message,
                "date_created": alert.date_created.isoformat(),
                "is_active": alert.is_active,
                "material_id": alert.material_id,
                "material_name": alert.material.material_name if alert.material else None,
                "project_name": alert.material.project.name if alert.material and alert.material.project else None
            }
            alerts_data.append(alert_data)
        
        return alerts_data
        
    except Exception as e:
        logger.exception("Error fetching alerts")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to fetch alerts"
        )

@app.post("/api/alerts/{alert_id}/resolve", response_class=JSONResponse, tags=["Alerts"])
async def api_resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    """Mark an alert as resolved (inactive)."""
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.is_active = False
        db.commit()
        
        return {"message": "Alert resolved successfully", "alert_id": alert_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error resolving alert")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to resolve alert"
        )

# === Agent API Endpoints ===
@app.get("/api/debris/report", response_class=JSONResponse, tags=["AI Agents"])
async def get_debris_analysis_report():
    """ Triggers the debris analysis agent and returns the report. """
    logger.info("Received request for debris analysis report.")
    try:
        report = run_debris_analysis_agent()
        logger.info("Debris analysis report generated successfully.")
        return JSONResponse(content={"report": report})
    except Exception as e:
        logger.exception("Error running debris analysis agent")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate debris report: {str(e)}")

@app.get("/api/inventory/report", response_class=JSONResponse, tags=["AI Agents"])
async def get_inventory_analysis_report():
    """ Triggers the inventory analysis agent and returns the report. """
    logger.info("Received request for inventory analysis report.")
    try:
        report = run_inventory_analysis_agent()
        logger.info("Inventory analysis report generated successfully.")
        return JSONResponse(content={"report": report})
    except Exception as e:
        logger.exception("Error running inventory analysis agent")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate inventory report: {str(e)}")

@app.get("/api/reports/resource-limits", response_class=JSONResponse, tags=["Reports"])
async def api_get_resource_limits_report(project_id: int, db: Session = Depends(get_db)):
    try:
        report = get_resource_limits_report(db, project_id)
        return JSONResponse(content=report)
    except Exception as e:
        logger.exception(f"Error generating resource limits report for project {project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate resource limits report.")

@app.get("/api/ai/procurement", response_class=JSONResponse, tags=["AI Agents"])
async def api_get_procurement_advice(project_id: int, lookback_days: int = 90):
    """
    Returns procurement suggestions and LLM explanations for the given project.
    Example: /api/ai/procurement?project_id=1&lookback_days=90
    """
    try:
        report = run_procurement_advisor(project_id, lookback_days=lookback_days)
        return JSONResponse(content=report)
    except Exception as e:
        logger.exception("Error running procurement advisor", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))