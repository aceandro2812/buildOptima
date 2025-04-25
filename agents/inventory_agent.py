# agents/inventory_agent.py (Absolute Imports, Pydantic V2, DDG Search)

import os
import json
import logging
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, TypedDict, Sequence, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun # Import DuckDuckGo tool

# --- Add parent directory to path for standalone execution ---
if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# --- Database Access (Using Absolute Imports) ---
from database import SessionLocal
from crud import (
    get_inventory as crud_get_inventory,
    get_consumption_data as crud_get_consumption_data,
    get_suppliers as crud_get_suppliers,
    get_cost_data as crud_get_cost_data # Optional cost data
)
from models import Inventory, Consumption, Supplier, Cost, Project # Import Project model
# For type hints if needed - Although crud functions return ORM objects

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Setup ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "mistralai/mistral-7b-instruct")
if not OPENROUTER_API_KEY: raise ValueError("OPENROUTER_API_KEY not found")

# --- LLM Configuration ---
try:
    llm = ChatOpenAI( model=OPENROUTER_MODEL_NAME, openai_api_key=OPENROUTER_API_KEY, openai_api_base="https://openrouter.ai/api/v1", temperature=0.5 )
    logging.info(f"LLM initialized with model: {OPENROUTER_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}")
    raise

# --- Tool Definitions ---
# Note: These Pydantic models are internal to the agent's tools
class InventoryRecord(BaseModel):
    id: int
    material_name: str
    quantity: float
    unit: str
    reorder_point: float
    last_updated: Optional[str] = None
    supplier_id: Optional[int] = None # Keep optional as per DB model
    supplier_name: Optional[str] = None
    project_id: int # Keep required as per DB model
    project_name: Optional[str] = None

class ConsumptionRecord(BaseModel):
    id: int
    material_id: int
    material_name: Optional[str] = None
    project: str # This will store the project *name*
    quantity_used: float
    date_used: Optional[str] = None
    notes: Optional[str] = None

class SupplierRecord(BaseModel):
    id: int
    name: str
    lead_time_days: Optional[int] = None # Keep optional
    reliability_rating: Optional[float] = None # Keep optional

class CostRecord(BaseModel):
    id: int
    material_id: int
    material_name: Optional[str] = None
    supplier_id: Optional[int] = None # Keep optional
    supplier_name: Optional[str] = None
    unit_price: float
    quantity_purchased: float
    total_cost: Optional[float] = None
    date_recorded: Optional[str] = None


@tool
def get_inventory_data() -> List[InventoryRecord]:
    """ Fetches current inventory levels for all materials. """
    db = SessionLocal()
    result = []
    try:
        # Use crud function which eager loads relationships
        inventory_orm = crud_get_inventory(db)
        for item in inventory_orm:
            # Supplier and Project should be loaded by crud function
            supplier_name = item.supplier.name if item.supplier else "N/A"
            project_name = item.project.name if item.project else "N/A"

            # Check if project_id exists before creating record
            if item.project_id is None:
                 logging.warning(f"Skipping inventory record ID {item.id} due to missing project_id")
                 continue

            try:
                 record = InventoryRecord(
                     id=item.id,
                     material_name=item.material_name,
                     quantity=item.quantity,
                     unit=item.unit,
                     reorder_point=item.reorder_point,
                     last_updated=item.last_updated.isoformat() if item.last_updated else None,
                     supplier_id=item.supplier_id, # Can be None
                     supplier_name=supplier_name,
                     project_id=item.project_id, # Required
                     project_name=project_name
                 )
                 result.append(record)
            except Exception as pydantic_error:
                 logging.warning(f"Skipping inventory record ID {item.id} due to Pydantic validation error: {pydantic_error}")

        logging.info(f"Fetched {len(result)} inventory records.")
    except Exception as e:
        logging.error(f"Error fetching inventory data: {e}", exc_info=True)
    finally:
        db.close()
    return result

# --- Tool with FIX ---
@tool
def get_consumption_history(days_limit: int = 90) -> List[ConsumptionRecord]:
    """ Fetches material consumption records from the past specified number of days. """
    db = SessionLocal()
    result = []
    skipped_count = 0
    try:
        # Use crud function which eager loads relationships
        consumption_orm = crud_get_consumption_data(db)
        cutoff_date = datetime.now() - timedelta(days=days_limit)
        for item in consumption_orm:
            # Basic checks
            if item.material_id is None or item.project_id is None:
                 logging.warning(f"Skipping consumption record ID {item.id} due to missing material_id or project_id")
                 skipped_count += 1
                 continue
            if not (item.date_used and item.date_used >= cutoff_date.replace(tzinfo=item.date_used.tzinfo)): # Make tz aware if needed
                 continue

            # --- FIX: Access project name via project_rel ---
            project_name = item.project_rel.name if item.project_rel else "N/A"
            # --- END FIX ---
            material_name = item.material.material_name if item.material else "N/A"

            try:
                 # Create the Pydantic record using the correct project name
                 record = ConsumptionRecord(
                     id=item.id,
                     material_id=item.material_id,
                     material_name=material_name,
                     project=project_name, # Assign the fetched project name
                     quantity_used=item.quantity_used,
                     date_used=item.date_used.isoformat() if item.date_used else None,
                     notes=item.notes
                 )
                 result.append(record)
            except Exception as pydantic_error:
                 # Catch Pydantic errors during record creation specifically
                 logging.warning(f"Skipping consumption record ID {item.id} due to Pydantic validation error: {pydantic_error}")
                 skipped_count += 1

        if skipped_count > 0:
            logging.warning(f"Skipped {skipped_count} consumption records due to missing ID, validation errors, or date range.")
        logging.info(f"Fetched {len(result)} valid consumption records from last {days_limit} days.")
    except Exception as e:
        logging.error(f"Error fetching consumption data: {e}", exc_info=True)
    finally:
        db.close()
    return result
# --- End Tool with FIX ---

@tool
def get_supplier_details() -> List[SupplierRecord]:
    """ Fetches relevant details (ID, name, lead time, reliability) for all suppliers. """
    db = SessionLocal(); result = []
    try:
        suppliers_orm = crud_get_suppliers(db)
        result = [ SupplierRecord( id=item.id, name=item.name, lead_time_days=item.lead_time_days, reliability_rating=item.reliability_rating ) for item in suppliers_orm ]
        logging.info(f"Fetched {len(result)} supplier records.")
    except Exception as e: logging.error(f"Error fetching supplier data: {e}", exc_info=True)
    finally: db.close(); return result

@tool
def get_cost_history(days_limit: int = 180) -> List[CostRecord]:
    """ Fetches material cost records from the past specified number of days. """
    db = SessionLocal(); result = []; skipped_count = 0
    try:
        # Use crud function which eager loads relationships
        cost_orm = crud_get_cost_data(db)
        cutoff_date = datetime.now() - timedelta(days=days_limit)
        for item in cost_orm:
            # Basic checks
            if item.material_id is None: # supplier_id can be None in Cost model
                skipped_count += 1; continue
            if not (item.date_recorded and item.date_recorded >= cutoff_date.replace(tzinfo=item.date_recorded.tzinfo)): # Make tz aware if needed
                 continue

            material_name = item.material.material_name if item.material else "N/A"
            supplier_name = item.supplier.name if item.supplier else "N/A"

            try:
                record = CostRecord(
                     id=item.id,
                     material_id=item.material_id,
                     material_name=material_name,
                     supplier_id=item.supplier_id, # Can be None
                     supplier_name=supplier_name,
                     unit_price=item.unit_price,
                     quantity_purchased=item.quantity_purchased,
                     total_cost=item.total_cost, # Should be calculated and stored now
                     date_recorded=item.date_recorded.isoformat() if item.date_recorded else None
                )
                result.append(record)
            except Exception as pydantic_error:
                 logging.warning(f"Skipping cost record ID {item.id} due to Pydantic validation error: {pydantic_error}"); skipped_count += 1

        if skipped_count > 0: logging.warning(f"Skipped {skipped_count} cost records due to missing IDs, validation errors, or date range.")
        logging.info(f"Fetched {len(result)} valid cost records from last {days_limit} days.")
    except Exception as e: logging.error(f"Error fetching cost data: {e}", exc_info=True)
    finally: db.close(); return result

# Initialize the DuckDuckGo Search tool instance
search_tool = DuckDuckGoSearchRun()

# --- Agent State Definition ---
class InventoryAnalysisState(TypedDict):
    inventory_data: List[InventoryRecord]
    consumption_data: List[ConsumptionRecord]
    supplier_data: List[SupplierRecord]
    cost_data: Optional[List[CostRecord]] = None
    consumption_analysis: Optional[str] = None
    optimization_suggestions: Optional[str] = None
    risk_assessment: Optional[str] = None
    price_trends: Optional[str] = None
    final_report: Optional[str] = None
    error_message: Optional[str] = None
    messages: Sequence[BaseMessage]

# --- Agent Node Functions ---
def fetch_inventory_data_node(state: InventoryAnalysisState) -> InventoryAnalysisState:
    logging.info("--- Node: Fetching Inventory Data ---")
    try:
        inventory = get_inventory_data.invoke({})
        consumption = get_consumption_history.invoke({"days_limit": 90})
        suppliers = get_supplier_details.invoke({})
        costs = get_cost_history.invoke({"days_limit": 180})
        state['inventory_data'] = inventory
        state['consumption_data'] = consumption
        state['supplier_data'] = suppliers
        state['cost_data'] = costs
        state['error_message'] = None
        inv_summary = f"Inventory: {len(inventory)} items. "
        con_summary = f"Consumption: {len(consumption)} records (last 90d). "
        sup_summary = f"Suppliers: {len(suppliers)}. "
        cost_summary = f"Costs: {len(costs)} records (last 180d)."
        state['messages'] = [HumanMessage(content=f"Data fetched: {inv_summary}{con_summary}{sup_summary}{cost_summary}")]
        logging.info("All inventory-related data fetched.")
    except Exception as e:
        logging.error(f"Error in fetch_inventory_data_node: {e}", exc_info=True)
        state['error_message'] = f"Failed to fetch necessary data: {e}"
        state['inventory_data'] = []
        state['consumption_data'] = []
        state['supplier_data'] = []
        state['cost_data'] = []
        state['messages'] = [SystemMessage(content=f"Error fetching data: {e}")]
    return state


def analyze_consumption_demand_node(state: InventoryAnalysisState) -> InventoryAnalysisState:
    logging.info("--- Node: Analyzing Consumption & Demand ---")
    if state.get('error_message'): return state
    inventory = state.get('inventory_data', [])
    consumption = state.get('consumption_data', [])
    if not consumption:
        analysis = "Insufficient data for consumption analysis (No valid consumption records found in the specified period)."
        logging.warning(analysis); state['consumption_analysis'] = analysis
        if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=analysis)]; return state
    if not inventory: logging.warning("Inventory data is missing, consumption analysis might be less useful.")

    # Use the agent's internal ConsumptionRecord for summaries
    consumption_summary_for_llm = [ f"Material ID {c.material_id} ({c.material_name}): Used {c.quantity_used} for Project '{c.project}' on {c.date_used[:10] if c.date_used else 'N/A'}" for c in consumption[:50] ]
    # Use the agent's internal InventoryRecord for summaries
    inventory_summary_for_llm = [ f"Material ID {i.id} ({i.material_name}) in Project '{i.project_name}': Current Qty {i.quantity} {i.unit}" for i in inventory[:20] ]

    prompt = f"""
You are an inventory analyst for a construction site in Thane, India.
Analyze the provided recent consumption history (last 90 days) and current inventory levels.
Identify materials with high consumption rates across different projects.
Estimate the average monthly consumption for the top 3-5 most consumed materials based on the 90-day data.
Highlight any materials showing significant recent spikes or drops in usage.

Consumption Data Preview (up to 50 records):
{json.dumps(consumption_summary_for_llm, indent=2)}
Total Valid Consumption Records Analyzed (last 90d): {len(consumption)}

Current Inventory Preview (up to 20 items):
{json.dumps(inventory_summary_for_llm, indent=2)}
Total Inventory Items: {len(inventory)}

Provide a concise analysis focusing on consumption trends and estimated monthly demand for key items. Mention projects if consumption is project-specific.
"""
    messages = [SystemMessage(content=prompt)]
    try:
        response = llm.invoke(messages); analysis = response.content
        state['consumption_analysis'] = analysis
        if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
        state['messages'] = state['messages'] + [response]
        logging.info("Consumption analysis generated.")
    except Exception as e:
        logging.error(f"LLM invocation failed during consumption analysis: {e}", exc_info=True)
        state['error_message'] = f"LLM error during consumption analysis: {e}"; state['consumption_analysis'] = "Error during consumption analysis."
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during consumption analysis: {e}")]
    return state


# --- Updated optimize_inventory_node ---
def optimize_inventory_node(state: InventoryAnalysisState) -> InventoryAnalysisState:
    """Suggests optimized reorder points or order quantities, incorporating real web search for price context."""
    logging.info("--- Node: Optimizing Inventory Levels ---")
    if state.get('error_message'): return state
    if not state.get('consumption_analysis') or "Insufficient data" in state.get('consumption_analysis', "") or "No valid consumption records" in state.get('consumption_analysis', "") :
        suggestions = "Skipping optimization due to lack of consumption analysis."
        logging.warning(suggestions); state['optimization_suggestions'] = suggestions
        if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=suggestions)]; return state
    inventory = state.get('inventory_data', []); suppliers = state.get('supplier_data', []); consumption_analysis = state.get('consumption_analysis', "No analysis available.")
    if not inventory or not suppliers:
         suggestions = "Skipping optimization due to missing inventory or supplier data."
         logging.warning(suggestions); state['optimization_suggestions'] = suggestions
         if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
         state['messages'] = state['messages'] + [SystemMessage(content=suggestions)]; return state

    # Use the agent's internal InventoryRecord for details
    inventory_details = [ f"ID {i.id} ({i.material_name}) in Project '{i.project_name}': Qty={i.quantity}, Unit={i.unit}, ReorderPt={i.reorder_point}, SupplierID={i.supplier_id or 'N/A'}" for i in inventory ]
    supplier_details = [ f"ID {s.id} ({s.name}): LeadTime={s.lead_time_days or 'N/A'}d, Reliability={s.reliability_rating or 'N/A'}/5" for s in suppliers ]
    top_material_search_query = None
    price_context = "No current price search performed or material not identified." # Default message

    # Attempt to parse top consumed material from analysis for search query
    try:
        lines = consumption_analysis.split('\n')
        for line in lines:
             match = None
             if "average monthly consumption" in line.lower() and ":" in line:
                 # Try to find name within potential parentheses first
                 parts = line.split('(')
                 material_name_candidate = parts[1].split(')')[0].strip() if len(parts) > 1 else None
                 if material_name_candidate and not material_name_candidate.isdigit():
                     match = material_name_candidate
                 # Fallback: try to get name before colon
                 if not match:
                     material_name_candidate = line.split(':')[0].strip()
                     # Basic check to avoid picking up numbers or short labels
                     if material_name_candidate and not material_name_candidate.isdigit() and len(material_name_candidate.split()) > 0:
                         # Remove potential leading characters like '*' or '-'
                         match = material_name_candidate.lstrip('*- ').strip()

                 if match:
                     # Refine query for better price results
                     top_material_search_query = f"price per kg OR price per cubic meter OR price per piece for construction material {match} in Thane India market"
                     logging.info(f"Identified potential top material for search: {match}")
                     break # Found the first likely material
    except Exception as parse_error:
        logging.warning(f"Could not parse top material from consumption analysis for price search: {parse_error}")

    # *** Perform Actual DuckDuckGo Search ***
    if top_material_search_query:
        logging.info(f"Performing DDG search for: {top_material_search_query}")
        try:
            # Use the initialized search_tool directly
            price_search_results = search_tool.run(top_material_search_query)
            # Limit result length for context window
            price_context = f"Recent price context search results for '{match}' in Thane (Top results):\n{price_search_results[:1000]}..." # Limit to 1000 chars
            logging.info(f"DDG search completed for {match}.")
            state['price_trends'] = price_context # Store search result/summary
        except Exception as e:
            logging.error(f"DuckDuckGo search failed: {e}", exc_info=True)
            price_context = f"Failed to perform price search for {match}. Error: {e}"
            state['price_trends'] = price_context # Store error message
    # *** End Search Logic ***

    # Prepare prompt for LLM, including the actual price context
    prompt = f"""
You are an inventory optimization specialist for a construction site in Thane.
Based on the consumption analysis, current inventory (potentially across multiple projects), supplier lead times, and recent price context (if available), suggest inventory adjustments.

Consumption Analysis Highlights:
{consumption_analysis}

Current Inventory (Note: Items are project-specific):
{json.dumps(inventory_details, indent=2)}

Supplier Lead Times & Reliability:
{json.dumps(supplier_details, indent=2)}

{price_context}

For materials identified as high-consumption or nearing reorder points:
1.  Identify the specific inventory item (Material Name + Project Name) that needs attention.
2.  Calculate the 'safety stock' needed for THAT SPECIFIC ITEM. Use average daily consumption (estimated from monthly consumption / 30) * lead time (from supplier details, use average if multiple suppliers for same material type exist, or state assumption) * reliability factor (e.g., 1.0 for 5/5, 1.2 for 4/5, 1.5 for <4/5 or N/A). If lead time or reliability is missing ('N/A'), state that calculation is approximate.
3.  Recommend adjustments to the 'reorder_point' for THAT SPECIFIC ITEM if the current one seems too low based on safety stock + lead time demand (demand during lead time = avg daily consumption * lead time).
4.  Suggest optimal 'order quantity' for THAT SPECIFIC ITEM, considering estimated monthly demand, current stock, and maybe recent price trends (mention if prices seem high/low based on search). Aim for roughly 1-1.5 months of stock after ordering for that item.
5.  Flag items significantly below their *calculated* reorder point (current qty < calculated reorder point), clearly stating the material name and project.

Provide concise, actionable suggestions for 3-5 key inventory items (Material + Project). Ensure calculations are shown or explained. Write in natural, professional language.
"""
    messages = [SystemMessage(content=prompt)]
    try:
        response = llm.invoke(messages); suggestions = response.content
        state['optimization_suggestions'] = suggestions
        if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
        state['messages'] = state['messages'] + [response]
        logging.info("Inventory optimization suggestions generated.")
    except Exception as e:
        logging.error(f"LLM invocation failed during optimization: {e}", exc_info=True)
        state['error_message'] = f"LLM error during optimization: {e}"; state['optimization_suggestions'] = "Error during optimization."
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during optimization: {e}")]
    return state
# --- End Updated optimize_inventory_node ---


def assess_risks_node(state: InventoryAnalysisState) -> InventoryAnalysisState:
    logging.info("--- Node: Assessing Inventory Risks ---")
    if state.get('error_message'): return state
    if not state.get('optimization_suggestions') or "Skipping optimization" in state.get('optimization_suggestions',''):
        assessment = "Skipping risk assessment as optimization suggestions are unavailable."
        logging.warning(assessment); state['risk_assessment'] = assessment
        if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=assessment)]; return state

    optimization_suggestions = state['optimization_suggestions']
    inventory = state.get('inventory_data', []) # Get inventory data for context

    inventory_context = "\n".join([f"- {i.material_name} (Project: {i.project_name}): Current Qty={i.quantity}, Reorder Pt={i.reorder_point}" for i in inventory])

    prompt = f"""
Based on the inventory optimization suggestions provided below, and the current inventory context, explicitly list any materials flagged as being:
1.  At immediate risk of stockout (significantly below *calculated* reorder point as mentioned in suggestions). Clearly state Material Name and Project Name.
2.  Potentially overstocked (e.g., having much more than 2-3 months of estimated demand on hand, based on suggestions or inventory levels compared to consumption trends). Clearly state Material Name and Project Name.

Optimization Suggestions:
{optimization_suggestions}

Current Inventory Context:
{inventory_context}

List the risks clearly using natural, professional language and bullet points. If no specific risks were flagged in the suggestions or evident from context, state "No immediate risks identified based on the analysis."
"""
    messages = [SystemMessage(content=prompt)]
    try:
        response = llm.invoke(messages); assessment = response.content
        state['risk_assessment'] = assessment
        if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
        state['messages'] = state['messages'] + [response]
        logging.info("Inventory risk assessment generated.")
    except Exception as e:
        logging.error(f"LLM invocation failed during risk assessment: {e}", exc_info=True)
        state['error_message'] = f"LLM error during risk assessment: {e}"; state['risk_assessment'] = "Error during risk assessment."
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during risk assessment: {e}")]
    return state


def compile_inventory_report_node(state: InventoryAnalysisState) -> InventoryAnalysisState:
    logging.info("--- Node: Compiling Inventory Report ---")
    if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
    if state.get('error_message') and not state.get('final_report'):
       state['final_report'] = f"## Report Error\n\nInventory report generation incomplete due to an error:\n\n* **Error:** {state['error_message']}"
       logging.warning("Compiling inventory report based on error state.")
       state['messages'] = state['messages'] + [SystemMessage(content="Inventory report compiled with errors.")]
       return state

    # Extract data summaries for the final report prompt
    inv_data_summary = json.dumps([i.model_dump(exclude={'messages'}) for i in state.get('inventory_data', [])[:5]], indent=2)
    cons_data_summary = json.dumps([c.model_dump(exclude={'messages'}) for c in state.get('consumption_data', [])[:5]], indent=2)
    cost_data_summary = json.dumps([c.model_dump(exclude={'messages'}) for c in state.get('cost_data', [])[:5]], indent=2)

    summary_prompt = f"""
Compile a professional inventory analysis report for a construction site manager in Thane, India.

**Instructions:**
1.  **Structure:** Use the following Markdown H2 headings ONLY: `## Executive Summary`, `## Consumption & Demand Analysis`, `## Optimization Suggestions`, `## Risk Assessment`, `## Price Context`.
2.  **Tone:** Write in clear, concise, and professional natural language. Avoid jargon where possible.
3.  **Formatting:**
    * Use standard paragraphs for explanations.
    * Use bullet points (`*` or `-`) for lists (like summary points, suggestions, or risks).
    * **IMPORTANT:** Do NOT use markdown bolding (`**text**`) simply to create labels within sentences (e.g., avoid "**Material:** Steel"). Instead, write naturally (e.g., "Analysis indicates steel consumption is high..."). Use bolding only for emphasis where appropriate in standard writing.
4.  **Content:** Synthesize the provided analysis information under the correct headings. Start with a brief Executive Summary (2-4 key takeaways covering main findings like high consumption items, risks, or key optimization suggestions). If analysis steps were skipped due to lack of data, mention this appropriately in the relevant sections. Incorporate project details where relevant (e.g., when mentioning specific inventory items or consumption).

**Available Analysis Information:**
Consumption Analysis Info:
{state.get('consumption_analysis', 'Analysis could not be performed due to missing data.')}

Optimization Suggestions Info:
{state.get('optimization_suggestions', 'Optimization could not be performed.')}

Risk Assessment Info:
{state.get('risk_assessment', 'Risk assessment could not be performed.')}

Price Trend Context Info:
{state.get('price_trends', 'No price search performed or search failed.')}

**Raw Data Snippets (for context, do not reproduce directly in report):**
Inventory Preview: {inv_data_summary} ...
Consumption Preview: {cons_data_summary} ...
Cost Preview: {cost_data_summary} ...

Generate the final report following these instructions precisely. Add a timestamp and location context at the beginning, and a concluding note about data limitations/estimates at the end.
"""
    messages = [SystemMessage(content=summary_prompt)]
    try:
        response = llm.invoke(messages); report_content = response.content
        timestamp = f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nLocation Context: Thane, Maharashtra, India\n\n"
        if not report_content.strip().startswith("Report Generated:"): report_content = timestamp + report_content
        if "Note:" not in report_content[-200:]: report_content += "\n\n---\n*Note: This report uses AI analysis based on available data. Calculations are estimates. Verify suggestions before placing orders.*"
        state['final_report'] = report_content.strip()
        if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
        state['messages'] = state['messages'] + [response, SystemMessage(content="Inventory report compiled successfully.")]
        logging.info("Inventory report compiled successfully using LLM with natural language instructions.")
    except Exception as e:
        logging.error(f"LLM invocation failed during inventory report compilation: {e}", exc_info=True)
        state['error_message'] = f"LLM error during inventory report compilation: {e}"; state['final_report'] = f"## Report Error\n\nFailed to compile inventory report using LLM due to an error:\n\n* **Error:** {e}"
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during inventory report compilation: {e}")]
    return state


# --- Build the Graph ---
logging.info("Building the Inventory Agent workflow...")
workflow = StateGraph(InventoryAnalysisState)
workflow.add_node("fetch_data", fetch_inventory_data_node)
workflow.add_node("analyze_consumption", analyze_consumption_demand_node)
workflow.add_node("optimize_inventory", optimize_inventory_node)
workflow.add_node("assess_risks", assess_risks_node)
workflow.add_node("compile_report", compile_inventory_report_node)
workflow.set_entry_point("fetch_data")
workflow.add_edge("fetch_data", "analyze_consumption")
workflow.add_edge("analyze_consumption", "optimize_inventory")
workflow.add_edge("optimize_inventory", "assess_risks")
workflow.add_edge("assess_risks", "compile_report")
workflow.add_edge("compile_report", END)
try:
    inventory_agent_graph = workflow.compile()
    logging.info("Inventory Agent workflow compiled successfully.")
except Exception as e:
    logging.error(f"Failed to compile Inventory Agent workflow: {e}")
    raise


# --- Function to Invoke the Agent ---
def run_inventory_analysis_agent() -> str:
    logging.info("Starting Inventory Analysis Agent workflow...")
    # Initialize state correctly according to the TypedDict definition
    initial_state = InventoryAnalysisState(
        inventory_data=[],
        consumption_data=[],
        supplier_data=[],
        cost_data=None,
        consumption_analysis=None,
        optimization_suggestions=None,
        risk_assessment=None,
        price_trends=None,
        final_report=None,
        error_message=None,
        messages=[]
    )
    try:
        config = {"recursion_limit": 15}
        # Ensure state is passed correctly if invoke expects a dictionary
        final_state = inventory_agent_graph.invoke(initial_state, config=config)
        logging.info("Inventory Analysis Agent workflow finished.")
        # Check for error message first if report compilation failed
        if final_state.get('error_message') and not final_state.get('final_report'):
             return f"Agent finished with error: {final_state['error_message']}"
        # Return final report or a default error message
        return final_state.get("final_report", "Error: Final report not found in agent state.")
    except Exception as e:
        logging.error(f"Exception during inventory agent graph invocation: {e}", exc_info=True)
        # Return a more informative error message to the API layer
        return f"Critical Error during inventory agent execution: {str(e)}"


# --- Direct Execution Example ---
if __name__ == "__main__":
    print("Running Inventory Agent Standalone Test...")
    # Make sure the database is populated with some data first for a meaningful test
    print("Ensure database (construction_materials.db) has relevant data.")
    report = run_inventory_analysis_agent()
    print("\n--- FINAL INVENTORY REPORT ---")
    print(report)