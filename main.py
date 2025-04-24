from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from database import get_db, engine
import models
from schemas import *
from crud import *
from crud import get_waste_data, create_waste_record, get_inventory # Import waste CRUD functions and get_inventory
from fastapi.responses import HTMLResponse, JSONResponse # Ensure JSONResponse is imported
from agents.debris_agent import run_debris_analysis_agent
from agents.inventory_agent import run_inventory_analysis_agent
# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Construction Material Manager")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# HOME PAGE
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    inventory = get_inventory(db)
    alerts = get_alerts(db)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "inventory": inventory, "alerts": alerts}
    )

# SUPPLIERS PAGE
@app.get("/suppliers", response_class=HTMLResponse)
async def suppliers_page(request: Request, db: Session = Depends(get_db)):
    suppliers = get_suppliers(db)
    return templates.TemplateResponse(
        "suppliers.html",
        {"request": request, "suppliers": suppliers}
    )

# CONSUMPTION PAGE
@app.get("/consumption", response_class=HTMLResponse)
async def consumption_page(request: Request, db: Session = Depends(get_db)):
    consumption_data = get_consumption_data(db)
    materials = get_inventory(db)
    return templates.TemplateResponse(
        "consumption.html",
        {"request": request, "consumption_data": consumption_data, "materials": materials}
    )

# COSTS PAGE
@app.get("/costs", response_class=HTMLResponse)
async def costs_page(request: Request, db: Session = Depends(get_db)):
    cost_data = get_cost_data(db)
    materials = get_inventory(db)
    suppliers = get_suppliers(db)
    return templates.TemplateResponse(
        "costs.html",
        {"request": request, "cost_data": cost_data, "materials": materials, "suppliers": suppliers}
    )

# MATERIALS PAGE
@app.get("/materials", response_class=HTMLResponse)
async def materials_page(request: Request, db: Session = Depends(get_db)):
    materials = get_inventory(db)
    suppliers = get_suppliers(db)
    return templates.TemplateResponse(
        "materials.html",
        {"request": request, "materials": materials, "suppliers": suppliers}
    )

# API ROUTES

# Inventory APIs
@app.get("/api/inventory", response_model=list[InventoryRead])
async def api_inventory(db: Session = Depends(get_db)):
    return get_inventory(db)

@app.post("/api/inventory", response_model=InventoryRead)
async def create_inventory_item(item: InventoryCreate, db: Session = Depends(get_db)):
    return create_inventory(db, item)

# Added endpoint to support the form action in materials.html
@app.post("/api/materials", response_model=InventoryRead)
async def create_material(material: InventoryCreate, db: Session = Depends(get_db)):
    return create_inventory(db, material)

@app.get("/api/materials/{material_id}")
async def get_material(material_id: int, db: Session = Depends(get_db)):
    material = get_inventory_item(db, material_id)
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    return material

@app.put("/api/materials/{material_id}")
async def update_material(material_id: int, material: InventoryCreate, db: Session = Depends(get_db)):
    updated = update_inventory_item(db, material_id, material)
    if not updated:
        raise HTTPException(status_code=404, detail="Material not found")
    return updated

@app.delete("/api/materials/{material_id}")
async def delete_material(material_id: int, db: Session = Depends(get_db)):
    success = delete_inventory_item(db, material_id)
    if not success:
        raise HTTPException(status_code=404, detail="Material not found")
    return {"status": "success"}

# Suppliers APIs
@app.get("/api/suppliers", response_model=list[SupplierRead])
async def api_suppliers(db: Session = Depends(get_db)):
    return get_suppliers(db)

@app.post("/api/suppliers", response_model=SupplierRead)
async def create_supplier(supplier: SupplierCreate, db: Session = Depends(get_db)):
    return create_supplier_db(db, supplier)

@app.get("/api/suppliers/{supplier_id}")
async def get_supplier(supplier_id: int, db: Session = Depends(get_db)):
    supplier = get_supplier_by_id(db, supplier_id)
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return supplier

@app.put("/api/suppliers/{supplier_id}")
async def update_supplier(supplier_id: int, supplier: SupplierCreate, db: Session = Depends(get_db)):
    updated = update_supplier_db(db, supplier_id, supplier)
    if not updated:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return updated

@app.delete("/api/suppliers/{supplier_id}")
async def delete_supplier(supplier_id: int, db: Session = Depends(get_db)):
    success = delete_supplier_db(db, supplier_id)
    if not success:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return {"status": "success"}

# Consumption APIs
@app.get("/api/consumption", response_model=list[ConsumptionRead])
async def api_consumption(db: Session = Depends(get_db)):
    return get_consumption_data(db)

@app.post("/api/consumption", response_model=ConsumptionRead)
async def create_consumption(consumption: ConsumptionCreate, db: Session = Depends(get_db)):
    return create_consumption_record(db, consumption)

# Costs APIs
@app.post("/api/costs", response_model=CostRead)
async def create_cost(cost: CostCreate, db: Session = Depends(get_db)):
    return create_cost_record(db, cost)



# Add this new route for the waste page
@app.get("/waste", response_class=HTMLResponse)
async def waste_page(request: Request, db: Session = Depends(get_db)):
    waste_records = get_waste_data(db)
    materials = get_inventory(db) # Need materials for the dropdown in the form
    return templates.TemplateResponse(
        "waste.html",
        {"request": request, "waste_records": waste_records, "materials": materials}
    )

# Add these new API routes for waste

@app.get("/api/waste", response_model=list[WasteRead])
async def api_get_waste(db: Session = Depends(get_db)):
    """
    Retrieve all waste records.
    """
    return get_waste_data(db)

@app.post("/api/waste", response_model=WasteRead, status_code=201)
async def api_create_waste(waste: WasteCreate, db: Session = Depends(get_db)):
    """
    Create a new waste record.
    Also updates the inventory quantity.
    """
    try:
        # The create_waste_record function already handles inventory update
        new_waste_record = create_waste_record(db, waste)
        return new_waste_record
    except Exception as e:
        # Basic error handling
        raise HTTPException(status_code=400, detail=str(e))
    
    
    
    

@app.get("/api/debris/report", response_class=JSONResponse)
async def get_debris_analysis_report():
    """
    Triggers the debris analysis agent and returns the generated report.
    """
    try:
        # Run the agent function we created
        # This function handles its own DB session internally for now
        report = run_debris_analysis_agent()
        return JSONResponse(content={"report": report})
    except Exception as e:
        # Log the exception for debugging
        # import logging
        # logging.exception("Error running debris analysis agent") # Optional: add logging
        raise HTTPException(status_code=500, detail=f"Failed to generate debris report: {str(e)}")




# --- ADD NEW INVENTORY AGENT ENDPOINT ---
@app.get("/api/inventory/report", response_class=JSONResponse)
async def get_inventory_analysis_report():
    """
    Triggers the inventory analysis agent and returns the generated report.
    """
    # logging.info("Received request for inventory analysis report.") # Added logging
    try:
        # Run the inventory agent function
        report = run_inventory_analysis_agent()
        # logging.info("Inventory analysis report generated successfully.") # Added logging
        return JSONResponse(content={"report": report})
    except Exception as e:
        # logging.exception("Error running inventory analysis agent") # Log full tracebac   k
        raise HTTPException(status_code=500, detail=f"Failed to generate inventory report: {str(e)}")

# --- Pydantic V2 Config for ORM Mode ---
# Add this configuration to your schemas if not already present
# In schemas.py, for each Read schema (e.g., InventoryRead, SupplierRead):
# class Config:
#     from_attributes = True # Replaces orm_mode = True for Pydantic V2
