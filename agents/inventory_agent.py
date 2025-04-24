# agents/inventory_agent.py (Absolute Imports, Pydantic V2, Natural Language Prompt, Syntax Fix)

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
from models import Inventory, Consumption, Supplier, Cost # For type hints if needed

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
class InventoryRecord(BaseModel):
    id: int; material_name: str; quantity: float; unit: str; reorder_point: float; last_updated: Optional[str] = None; supplier_id: int; supplier_name: Optional[str] = None
class ConsumptionRecord(BaseModel):
    id: int; material_id: int; material_name: Optional[str] = None; project: str; quantity_used: float; date_used: Optional[str] = None; notes: Optional[str] = None
class SupplierRecord(BaseModel):
    id: int; name: str; lead_time_days: int; reliability_rating: float
class CostRecord(BaseModel):
    id: int; material_id: int; material_name: Optional[str] = None; supplier_id: int; supplier_name: Optional[str] = None; unit_price: float; quantity_purchased: float; total_cost: Optional[float] = None; date_recorded: Optional[str] = None

@tool
def get_inventory_data() -> List[InventoryRecord]:
    """ Fetches current inventory levels for all materials. Skips items with missing supplier_id. """
    db = SessionLocal(); result = []
    try:
        inventory_orm = crud_get_inventory(db)
        result = [ InventoryRecord( id=item.id, material_name=item.material_name, quantity=item.quantity, unit=item.unit, reorder_point=item.reorder_point, last_updated=item.last_updated.isoformat() if item.last_updated else None, supplier_id=item.supplier_id, supplier_name=item.supplier.name if item.supplier else "N/A" ) for item in inventory_orm if item.supplier_id is not None ]
        logging.info(f"Fetched {len(result)} inventory records.")
    except Exception as e: logging.error(f"Error fetching inventory data: {e}", exc_info=True)
    finally: db.close(); return result

@tool
def get_consumption_history(days_limit: int = 90) -> List[ConsumptionRecord]:
    """ Fetches material consumption records from the past specified number of days. Skips records with missing material_id. """
    db = SessionLocal(); result = []; skipped_count = 0
    try:
        consumption_orm = crud_get_consumption_data(db)
        cutoff_date = datetime.now() - timedelta(days=days_limit)
        for item in consumption_orm:
            if item.material_id is None: skipped_count += 1; continue
            if not (item.date_used and item.date_used >= cutoff_date): continue
            try: result.append( ConsumptionRecord( id=item.id, material_id=item.material_id, material_name=item.material.material_name if item.material else "N/A", project=item.project, quantity_used=item.quantity_used, date_used=item.date_used.isoformat() if item.date_used else None, notes=item.notes ) )
            except Exception as pydantic_error: logging.warning(f"Skipping consumption record ID {item.id} due to Pydantic validation error: {pydantic_error}"); skipped_count += 1
        if skipped_count > 0: logging.warning(f"Skipped {skipped_count} consumption records due to missing ID or validation errors.")
        logging.info(f"Fetched {len(result)} valid consumption records from last {days_limit} days.")
    except Exception as e: logging.error(f"Error fetching consumption data: {e}", exc_info=True)
    finally: db.close(); return result

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
    """ Fetches material cost records from the past specified number of days. Skips records with missing material_id or supplier_id. """
    db = SessionLocal(); result = []; skipped_count = 0
    try:
        cost_orm = crud_get_cost_data(db)
        cutoff_date = datetime.now() - timedelta(days=days_limit)
        for item in cost_orm:
            if item.material_id is None or item.supplier_id is None: skipped_count += 1; continue
            if not (item.date_recorded and item.date_recorded >= cutoff_date): continue
            try: result.append( CostRecord( id=item.id, material_id=item.material_id, material_name=item.material.material_name if item.material else "N/A", supplier_id=item.supplier_id, supplier_name=item.supplier.name if item.supplier else "N/A", unit_price=item.unit_price, quantity_purchased=item.quantity_purchased, total_cost=item.total_cost, date_recorded=item.date_recorded.isoformat() if item.date_recorded else None ) )
            except Exception as pydantic_error: logging.warning(f"Skipping cost record ID {item.id} due to Pydantic validation error: {pydantic_error}"); skipped_count += 1
        if skipped_count > 0: logging.warning(f"Skipped {skipped_count} cost records due to missing IDs or validation errors.")
        logging.info(f"Fetched {len(result)} valid cost records from last {days_limit} days.")
    except Exception as e: logging.error(f"Error fetching cost data: {e}", exc_info=True)
    finally: db.close(); return result

search_tool = DuckDuckGoSearchRun()

# --- Agent State Definition ---
class InventoryAnalysisState(TypedDict):
    inventory_data: List[InventoryRecord]; consumption_data: List[ConsumptionRecord]; supplier_data: List[SupplierRecord]; cost_data: Optional[List[CostRecord]] = None
    consumption_analysis: Optional[str] = None; optimization_suggestions: Optional[str] = None; risk_assessment: Optional[str] = None; price_trends: Optional[str] = None
    final_report: Optional[str] = None; error_message: Optional[str] = None; messages: Sequence[BaseMessage]

# --- Agent Node Functions ---
def fetch_inventory_data_node(state: InventoryAnalysisState) -> InventoryAnalysisState:
    logging.info("--- Node: Fetching Inventory Data ---")
    try:
        inventory = get_inventory_data.invoke({})
        consumption = get_consumption_history.invoke({"days_limit": 90})
        suppliers = get_supplier_details.invoke({})
        costs = get_cost_history.invoke({"days_limit": 180})
        state['inventory_data'] = inventory; state['consumption_data'] = consumption; state['supplier_data'] = suppliers; state['cost_data'] = costs; state['error_message'] = None
        inv_summary = f"Inventory: {len(inventory)} items. "; con_summary = f"Consumption: {len(consumption)} records (last 90d). "; sup_summary = f"Suppliers: {len(suppliers)}. "; cost_summary = f"Costs: {len(costs)} records (last 180d)."
        state['messages'] = [HumanMessage(content=f"Data fetched: {inv_summary}{con_summary}{sup_summary}{cost_summary}")]
        logging.info("All inventory-related data fetched.")
    except Exception as e:
        logging.error(f"Error in fetch_inventory_data_node: {e}", exc_info=True)
        state['error_message'] = f"Failed to fetch necessary data: {e}"
        state['inventory_data'] = []; state['consumption_data'] = []; state['supplier_data'] = []; state['cost_data'] = []
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
    consumption_summary_for_llm = [ f"Material ID {c.material_id} ({c.material_name}): Used {c.quantity_used} on {c.date_used[:10]}" for c in consumption[:50] ]
    inventory_summary_for_llm = [ f"Material ID {i.id} ({i.material_name}): Current Qty {i.quantity} {i.unit}" for i in inventory[:20] ]
    prompt = f""" You are an inventory analyst for a construction site in Thane, India. Analyze the provided recent consumption history (last 90 days) and current inventory levels. Identify materials with high consumption rates. Estimate the average monthly consumption for the top 3-5 most consumed materials based on the 90-day data. Highlight any materials showing significant recent spikes or drops in usage. Consumption Data Preview (up to 50 records):\n{json.dumps(consumption_summary_for_llm, indent=2)}\nTotal Valid Consumption Records Analyzed (last 90d): {len(consumption)}\n Current Inventory Preview (up to 20 items):\n{json.dumps(inventory_summary_for_llm, indent=2)}\nTotal Inventory Items: {len(inventory)}\n Provide a concise analysis focusing on consumption trends and estimated monthly demand for key items. """ # Keep prompt concise
    messages = [SystemMessage(content=prompt)]
    try:
        response = llm.invoke(messages); analysis = response.content
        state['consumption_analysis'] = analysis; state['messages'] = state['messages'] + [response]
        logging.info("Consumption analysis generated.")
    except Exception as e:
        logging.error(f"LLM invocation failed during consumption analysis: {e}", exc_info=True)
        state['error_message'] = f"LLM error during consumption analysis: {e}"; state['consumption_analysis'] = "Error during consumption analysis."
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during consumption analysis: {e}")]
    return state

def optimize_inventory_node(state: InventoryAnalysisState) -> InventoryAnalysisState:
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
    inventory_details = [ f"ID {i.id} ({i.material_name}): Qty={i.quantity}, Unit={i.unit}, ReorderPt={i.reorder_point}, SupplierID={i.supplier_id}" for i in inventory ]
    supplier_details = [ f"ID {s.id} ({s.name}): LeadTime={s.lead_time_days}d, Reliability={s.reliability_rating}/5" for s in suppliers ]
    top_material_search_query = None; price_context = "No current price search performed."
    try: # Attempt to parse material for price search
        lines = consumption_analysis.split('\n')
        for line in lines:
             match = None
             if "average monthly consumption" in line.lower() and ":" in line:
                 parts = line.split('('); material_name_candidate = parts[1].split(')')[0].strip() if len(parts) > 1 else None
                 if material_name_candidate and not material_name_candidate.isdigit(): match = material_name_candidate
                 # *** SYNTAX FIX APPLIED HERE ***
                 if not match:
                     material_name_candidate = line.split(':')[0].strip()
                     if material_name_candidate and not material_name_candidate.isdigit():
                         match = material_name_candidate
                 # *** END SYNTAX FIX ***
                 if match: top_material_search_query = f"current price per {match} unit for construction in Thane India"; logging.info(f"Identified potential top material for search: {match}"); break
    except Exception as parse_error: logging.warning(f"Could not parse top material from consumption analysis for price search: {parse_error}")
    if top_material_search_query:
        logging.info(f"Performing DDG search for: {top_material_search_query}")
        try: price_search_results = search_tool.run(top_material_search_query); price_context = f"Recent price context search for '{top_material_search_query}':\n{price_search_results}"; logging.info("DDG search completed."); state['price_trends'] = price_context
        except Exception as e: logging.error(f"DuckDuckGo search failed: {e}", exc_info=True); price_context = f"Failed to perform price search for {top_material_search_query}."; state['price_trends'] = price_context
    prompt = f""" You are an inventory optimization specialist for a construction site in Thane. Based on the consumption analysis, current inventory, and supplier lead times, suggest inventory adjustments. Consumption Analysis Highlights:\n{consumption_analysis}\n Current Inventory:\n{json.dumps(inventory_details, indent=2)}\n Supplier Lead Times & Reliability:\n{json.dumps(supplier_details, indent=2)}\n {price_context}\n For materials identified as high-consumption or nearing reorder points: 1. Calculate the 'safety stock' needed (e.g., average daily consumption * lead time * reliability factor). Average daily consumption can be estimated from monthly consumption / 30. Use a reliability factor (e.g., 1.0 for 5/5 rating, 1.2 for 4/5, 1.5 for <4/5). If lead time or reliability is missing, state that calculation is approximate. 2. Recommend adjustments to the 'reorder_point' if the current one seems too low based on safety stock + lead time demand (demand during lead time = avg daily consumption * lead time). 3. Suggest optimal 'order quantity' considering estimated monthly demand, current stock, and maybe recent price trends (if available). Aim for roughly 1-1.5 months of stock after ordering. 4. Flag items significantly below the *calculated* reorder point (current qty < reorder point). Provide concise, actionable suggestions for 3-5 key materials. Ensure calculations are shown or explained. """ # Keep prompt concise
    messages = [SystemMessage(content=prompt)]
    try:
        response = llm.invoke(messages); suggestions = response.content
        state['optimization_suggestions'] = suggestions; state['messages'] = state['messages'] + [response]
        logging.info("Inventory optimization suggestions generated.")
    except Exception as e:
        logging.error(f"LLM invocation failed during optimization: {e}", exc_info=True)
        state['error_message'] = f"LLM error during optimization: {e}"; state['optimization_suggestions'] = "Error during optimization."
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during optimization: {e}")]
    return state

def assess_risks_node(state: InventoryAnalysisState) -> InventoryAnalysisState:
    logging.info("--- Node: Assessing Inventory Risks ---")
    if state.get('error_message'): return state
    if not state.get('optimization_suggestions') or "Skipping optimization" in state.get('optimization_suggestions',''):
        assessment = "Skipping risk assessment as optimization suggestions are unavailable."
        logging.warning(assessment); state['risk_assessment'] = assessment
        if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=assessment)]; return state
    optimization_suggestions = state['optimization_suggestions']
    prompt = f""" Based on the inventory optimization suggestions provided below, explicitly list any materials flagged as being: 1. At immediate risk of stockout (significantly below calculated reorder point). 2. Potentially overstocked (e.g., having much more than 2-3 months of estimated demand on hand). Optimization Suggestions:\n{optimization_suggestions}\n List the risks clearly. If no specific risks were flagged in the suggestions, state "No immediate risks identified based on the analysis." """ # Keep prompt concise
    messages = [SystemMessage(content=prompt)]
    try:
        response = llm.invoke(messages); assessment = response.content
        state['risk_assessment'] = assessment; state['messages'] = state['messages'] + [response]
        logging.info("Inventory risk assessment generated.")
    except Exception as e:
        logging.error(f"LLM invocation failed during risk assessment: {e}", exc_info=True)
        state['error_message'] = f"LLM error during risk assessment: {e}"; state['risk_assessment'] = "Error during risk assessment."
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during risk assessment: {e}")]
    return state

def compile_inventory_report_node(state: InventoryAnalysisState) -> InventoryAnalysisState:
    """Compiles the final inventory analysis report with a summary and structured sections, using natural language."""
    logging.info("--- Node: Compiling Inventory Report ---")
    if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
    if state.get('error_message') and not state.get('final_report'):
       state['final_report'] = f"## Report Error\n\nInventory report generation incomplete due to an error:\n\n* **Error:** {state['error_message']}"
       logging.warning("Compiling inventory report based on error state.")
       state['messages'] = state['messages'] + [SystemMessage(content="Inventory report compiled with errors.")]
       return state
    summary_prompt = f""" Compile a professional inventory analysis report for a construction site manager in Thane, India. **Instructions:** 1. **Structure:** Use the following Markdown H2 headings ONLY: `## Executive Summary`, `## Consumption & Demand Analysis`, `## Optimization Suggestions`, `## Risk Assessment`, `## Price Context`. 2. **Tone:** Write in clear, concise, and professional natural language. Avoid jargon where possible. 3. **Formatting:** * Use standard paragraphs for explanations. * Use bullet points (`*` or `-`) for lists (like summary points, suggestions, or risks). * **IMPORTANT:** Do NOT use markdown bolding (`**text**`) simply to create labels within sentences (e.g., avoid "**Material:** Steel"). Instead, write naturally (e.g., "Analysis indicates steel consumption is high..."). Use bolding only for emphasis where appropriate in standard writing. 4. **Content:** Synthesize the provided information under the correct headings. Start with a brief Executive Summary (2-4 key takeaways covering main findings like high consumption items, risks, or key optimization suggestions). If analysis steps were skipped due to lack of data, mention this appropriately in the relevant sections. **Available Information:** Consumption Analysis Info: {state.get('consumption_analysis', 'Analysis could not be performed due to missing data.')} Optimization Suggestions Info: {state.get('optimization_suggestions', 'Optimization could not be performed.')} Risk Assessment Info: {state.get('risk_assessment', 'Risk assessment could not be performed.')} Price Trend Context Info: {state.get('price_trends', 'No price search performed or search failed.')} Generate the final report following these instructions precisely. Add a timestamp and location context at the beginning, and a concluding note about data limitations/estimates at the end. """
    messages = [SystemMessage(content=summary_prompt)]
    try:
        response = llm.invoke(messages); report_content = response.content
        timestamp = f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nLocation Context: Thane, Maharashtra, India\n\n"
        if not report_content.strip().startswith("Report Generated:"): report_content = timestamp + report_content
        if "Note:" not in report_content[-200:]: report_content += "\n\n---\n*Note: This report uses AI analysis based on available data. Calculations are estimates. Verify suggestions before placing orders.*"
        state['final_report'] = report_content.strip(); state['messages'] = state['messages'] + [response, SystemMessage(content="Inventory report compiled successfully.")]
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
    initial_state = InventoryAnalysisState( inventory_data=[], consumption_data=[], supplier_data=[], cost_data=None, messages=[] )
    try:
        config = {"recursion_limit": 15}
        final_state = inventory_agent_graph.invoke(initial_state, config=config)
        logging.info("Inventory Analysis Agent workflow finished.")
        if final_state.get('error_message') and not final_state.get('final_report'):
             return f"Agent finished with error: {final_state['error_message']}"
        return final_state.get("final_report", "Error: Final report not found in agent state.")
    except Exception as e:
        logging.error(f"Exception during inventory agent graph invocation: {e}", exc_info=True)
        return f"Critical Error during inventory agent execution: {e}"

# --- Direct Execution Example ---
if __name__ == "__main__":
    print("Running Inventory Agent Standalone Test...")
    report = run_inventory_analysis_agent()
    print("\n--- FINAL INVENTORY REPORT ---")
    print(report)

