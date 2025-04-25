# agents/debris_agent.py (Absolute Imports, Pydantic V2, DDG Search)

import os
import json
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv
from typing import List, TypedDict, Sequence, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field # Use Pydantic V2 directly
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun # <-- Import DDG Tool

# --- Add parent directory to path for standalone execution ---
if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# --- Database Access (Using Absolute Imports) ---
from database import SessionLocal
from crud import get_waste_data as crud_get_waste_data
from models import Waste

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Setup ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "mistralai/mistral-7b-instruct")
if not OPENROUTER_API_KEY: raise ValueError("OPENROUTER_API_KEY not found")

# --- LLM Configuration ---
try:
    llm = ChatOpenAI( model=OPENROUTER_MODEL_NAME, openai_api_key=OPENROUTER_API_KEY, openai_api_base="https://openrouter.ai/api/v1", temperature=0.7 )
    logging.info(f"LLM initialized with model: {OPENROUTER_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}")
    raise

# --- Tool Definitions ---
class WasteRecord(BaseModel):
    id: int; material_name: str; project_name: str; quantity_wasted: float; unit: str; reason: Optional[str] = None; preventive_measures: Optional[str] = None; date_recorded: Optional[str] = None

@tool
def get_waste_database_records() -> List[WasteRecord]:
    """ Fetches all waste records... (Input is ignored). """
    db = SessionLocal(); result_list = []
    try:
        waste_records_orm = crud_get_waste_data(db)
        for record in waste_records_orm:
            material_info = {"material_name": "N/A", "unit": "N/A"}
            if record.material: material_info["material_name"] = record.material.material_name; material_info["unit"] = record.material.unit
            if record.material_id is None: logging.warning(f"Skipping waste record ID {record.id} due to missing material_id"); continue
            result_list.append(WasteRecord( id=record.id, material_name=material_info["material_name"], project_name=record.project_name, quantity_wasted=record.quantity_wasted, unit=material_info["unit"], reason=record.reason, preventive_measures=record.preventive_measures, date_recorded=record.date_recorded.isoformat() if record.date_recorded else None ))
        logging.info(f"Successfully fetched {len(result_list)} waste records from DB.")
    except Exception as e: logging.error(f"Error fetching waste data from DB: {e}", exc_info=True); return []
    finally: db.close(); return result_list

# Initialize the DuckDuckGo Search tool instance
search_tool = DuckDuckGoSearchRun() # <-- Initialize DDG Tool

# --- Agent State Definition ---
class DebrisAnalysisState(TypedDict):
    raw_waste_data: List[WasteRecord]; waste_summary: Optional[str]; disposal_options: Optional[str]; reduction_strategies: Optional[str]; final_report: Optional[str]; error_message: Optional[str]; messages: Sequence[BaseMessage]

# --- Agent Node Functions ---
def fetch_data_node(state: DebrisAnalysisState) -> DebrisAnalysisState:
    # ... (fetch_data_node remains the same) ...
    logging.info("--- Node: Fetching Waste Data ---")
    try:
        raw_data: List[WasteRecord] = get_waste_database_records.invoke({})
        state['raw_waste_data'] = raw_data
        data_preview = [record.model_dump(exclude={'id'}) for record in raw_data[:3]]
        state['messages'] = [HumanMessage( content=f"Raw waste data preview (first {len(data_preview)} records): {json.dumps(data_preview)}... Total records: {len(raw_data)}" )]
        logging.info(f"Fetched {len(raw_data)} records. Added preview to messages.")
        state['error_message'] = None
    except Exception as e:
        logging.error(f"Error in fetch_data_node: {e}", exc_info=True)
        state['error_message'] = f"Failed to fetch data: {e}"; state['raw_waste_data'] = []; state['messages'] = [SystemMessage(content=f"Error fetching waste data: {e}")]
    return state

def analyze_data_node(state: DebrisAnalysisState) -> DebrisAnalysisState:
    # ... (analyze_data_node remains the same) ...
    logging.info("--- Node: Analyzing Waste Data ---")
    if state.get('error_message'): logging.warning("Skipping analysis due to previous error."); return state
    raw_data = state.get('raw_waste_data', [])
    if not raw_data:
        summary = "No waste data found to analyze."
        logging.info(summary); state['waste_summary'] = summary
        if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=summary)]; return state
    last_message = state['messages'][-1] if state['messages'] else HumanMessage(content="No previous messages.")
    messages = [ SystemMessage(content="You are a construction waste analyst... (Based *only* on the provided data preview and total record count, provide a concise summary...)"), last_message ]
    try:
        response = llm.invoke(messages); summary = response.content
        state['waste_summary'] = summary; state['messages'] = state['messages'] + [response]
        logging.info("Waste data summary generated.")
    except Exception as e:
        logging.error(f"LLM invocation failed during analysis: {e}")
        state['error_message'] = f"LLM error during analysis: {e}"; state['waste_summary'] = "Error during analysis."
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during analysis: {e}")]
    return state

# --- Updated disposal_research_node ---
def disposal_research_node(state: DebrisAnalysisState) -> DebrisAnalysisState:
    """ Formulates a search query and executes it using DuckDuckGo to find disposal options. """
    logging.info("--- Node: Researching Disposal Options ---") # Renamed log slightly
    if state.get('error_message') or "No waste data" in state.get('waste_summary', ""):
        logging.warning("Skipping disposal research due to previous error or no data.")
        state['disposal_options'] = "Disposal research skipped."
        return state

    raw_data = state.get('raw_waste_data', [])
    location = "Thane, Maharashtra, India"
    # Get waste types, ensuring uniqueness and filtering N/A
    waste_types = list(set(item.material_name for item in raw_data if item.material_name and item.material_name != "N/A"))

    if not waste_types:
        options = "Could not determine specific waste types from data. Skipping disposal research."
        logging.warning(options)
        state['disposal_options'] = options
        if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=options)]
        return state

    # Formulate query using LLM based on summary and waste types
    last_message = state['messages'][-1] if state['messages'] else HumanMessage(content="No previous messages.")
    # Refined prompt for query formulation
    query_formulation_prompt = f"""
Based on the waste summary and identified waste types below, formulate a concise and effective web search query to find specific recycling facilities, disposal sites, associated costs (if possible), and eco-friendly options for construction waste in {location}. Focus on the most prominent waste types mentioned.

Identified waste types include: {', '.join(waste_types[:4])}...
Waste Summary Context: {state.get('waste_summary', 'N/A')}

Generate ONLY the search query string.
"""
    messages = [SystemMessage(content=query_formulation_prompt)] # Use only system prompt for direct query generation

    formulated_query = None
    try:
        response = llm.invoke(messages)
        formulated_query = response.content.strip().strip('"') # Clean up potential quotes
        logging.info(f"LLM formulated search query: {formulated_query}")
        state['messages'] = state['messages'] + [response] # Log LLM response

    except Exception as e:
        logging.error(f"LLM invocation failed during disposal query formulation: {e}", exc_info=True)
        state['error_message'] = f"LLM error during disposal research prep: {e}"
        state['disposal_options'] = "Error: Could not formulate search query."
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during disposal research prep: {e}")]
        return state # Stop if query formulation fails

    # *** Perform Actual DuckDuckGo Search ***
    if formulated_query:
        logging.info(f"Performing DDG search for: {formulated_query}")
        try:
            # Use the initialized search_tool directly
            search_results = search_tool.run(formulated_query)
            # Process/Store results (maybe summarize later if too long)
            disposal_info = f"Search results for '{formulated_query}' (Top results):\n{search_results[:1500]}..." # Limit length
            state['disposal_options'] = disposal_info
            state['messages'] = state['messages'] + [HumanMessage(content=f"DuckDuckGo Search Results (Truncated):\n{disposal_info}")]
            logging.info(f"DDG search completed for query: {formulated_query}")
        except Exception as e:
            logging.error(f"DuckDuckGo search failed: {e}", exc_info=True)
            error_msg = f"Error during DuckDuckGo search: {e}"
            state['disposal_options'] = f"Failed to perform search for query '{formulated_query}'. Error: {e}"
            # Decide if this is a critical error for the state
            # state['error_message'] = error_msg
            state['messages'] = state['messages'] + [SystemMessage(content=error_msg)]
    else:
        # Handle case where LLM failed to generate a query
        state['disposal_options'] = "Search skipped as query could not be formulated."
        logging.warning("Search skipped as query could not be formulated.")


    return state
# --- End Updated disposal_research_node ---


def reduction_strategy_node(state: DebrisAnalysisState) -> DebrisAnalysisState:
    # ... (reduction_strategy_node remains the same) ...
    logging.info("--- Node: Generating Reduction Strategies ---")
    if state.get('error_message') or "No waste data" in state.get('waste_summary', ""): logging.warning("Skipping reduction strategies..."); state['reduction_strategies'] = "Reduction strategies skipped."; return state
    last_message = state['messages'][-1] if state['messages'] else HumanMessage(content="No previous messages.")
    # Include disposal options context if available
    disposal_context = state.get('disposal_options', 'No disposal research performed.')
    messages = [ SystemMessage(content=f"You are a construction efficiency expert specializing in waste reduction in India. Based on the waste summary and potentially relevant disposal options found for Thane, suggest 3-5 practical, actionable, and locally relevant strategies to reduce construction waste. Consider material sourcing, handling, site practices, and potential reuse/recycling avenues mentioned.\n\nWaste Summary:\n{state.get('waste_summary')}\n\nDisposal Context:\n{disposal_context[:500]}..."), last_message ] # Add disposal context
    try:
        response = llm.invoke(messages); strategies = response.content
        state['reduction_strategies'] = strategies; state['messages'] = state['messages'] + [response]
        logging.info("Reduction strategies generated.")
    except Exception as e:
        logging.error(f"LLM invocation failed during strategy generation: {e}")
        state['error_message'] = f"LLM error during strategy generation: {e}"; state['reduction_strategies'] = "Error during strategy generation."
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during strategy generation: {e}")]
    return state


def compile_report_node(state: DebrisAnalysisState) -> DebrisAnalysisState:
    # ... (compile_report_node remains the same - already uses natural language prompt) ...
    logging.info("--- Node: Compiling Report ---")
    if 'messages' not in state or not isinstance(state['messages'], list): state['messages'] = []
    if state.get('error_message') and not state.get('final_report'):
       state['final_report'] = f"## Report Error\n\nReport generation incomplete due to an error:\n\n* **Error:** {state['error_message']}"
       logging.warning("Compiling report based on error state.")
       state['messages'] = state['messages'] + [SystemMessage(content="Final report compiled with errors.")]; return state
    summary_prompt = f""" Compile a professional waste analysis report for a construction site manager in Thane, India. **Instructions:** 1. **Structure:** Use the following Markdown H2 headings ONLY: `## Executive Summary`, `## Waste Analysis`, `## Disposal Options`, `## Reduction Strategies`. 2. **Tone:** Write in clear, concise, and professional natural language. Avoid jargon where possible. 3. **Formatting:** * Use standard paragraphs for explanations. * Use bullet points (`*` or `-`) for lists (like summary points or strategies). * **IMPORTANT:** Do NOT use markdown bolding (`**text**`) simply to create labels within sentences (e.g., avoid "**Material:** Steel"). Instead, write naturally (e.g., "The primary material found was steel..."). Use bolding only for emphasis where appropriate in standard writing. 4. **Content:** Synthesize the provided information under the correct headings. Start with a brief Executive Summary (2-3 key takeaways). If analysis steps were skipped or failed, mention this appropriately. **Available Information:** Waste Summary Info: {state.get('waste_summary', 'Analysis could not be performed.')} Disposal Options Info: {state.get('disposal_options', 'Research could not be performed or is pending.')} Reduction Strategies Info: {state.get('reduction_strategies', 'Suggestions could not be generated.')} Generate the final report following these instructions precisely. Add a timestamp and location context at the beginning, and a concluding note about data limitations/placeholders at the end. """
    messages = [SystemMessage(content=summary_prompt)]
    try:
        response = llm.invoke(messages); report_content = response.content
        timestamp = f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nLocation Context: Thane, Maharashtra, India\n\n"
        if not report_content.strip().startswith("Report Generated:"): report_content = timestamp + report_content
        if "Note:" not in report_content[-200:]: report_content += "\n\n---\n*Note: This report is based on available data and AI analysis. Disposal options require verification based on search results.*" # Updated note
        state['final_report'] = report_content.strip(); state['messages'] = state['messages'] + [response, SystemMessage(content="Final report compiled successfully.")]
        logging.info("Report compiled successfully using LLM with natural language instructions.")
    except Exception as e:
        logging.error(f"LLM invocation failed during report compilation: {e}", exc_info=True)
        state['error_message'] = f"LLM error during report compilation: {e}"; state['final_report'] = f"## Report Error\n\nFailed to compile report using LLM due to an error:\n\n* **Error:** {e}"
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during report compilation: {e}")]
    return state


# --- Build the Graph ---
# ... (Graph definition remains the same) ...
logging.info("Building the LangGraph workflow...")
workflow = StateGraph(DebrisAnalysisState)
workflow.add_node("fetch_data", fetch_data_node)
workflow.add_node("analyze_data", analyze_data_node)
workflow.add_node("research_disposal", disposal_research_node)
workflow.add_node("suggest_reduction", reduction_strategy_node)
workflow.add_node("compile_report", compile_report_node)
workflow.set_entry_point("fetch_data")
workflow.add_edge("fetch_data", "analyze_data")
workflow.add_edge("analyze_data", "research_disposal")
workflow.add_edge("research_disposal", "suggest_reduction")
workflow.add_edge("suggest_reduction", "compile_report")
workflow.add_edge("compile_report", END)
try:
    debris_agent_graph = workflow.compile()
    logging.info("LangGraph workflow compiled successfully.")
except Exception as e:
    logging.error(f"Failed to compile LangGraph workflow: {e}")
    raise


# --- Function to Invoke the Agent ---
# ... (Invocation function remains the same) ...
def run_debris_analysis_agent() -> str:
    logging.info("Starting Debris Analysis Agent workflow...")
    initial_state = DebrisAnalysisState( raw_waste_data=[], waste_summary=None, disposal_options=None, reduction_strategies=None, final_report=None, error_message=None, messages=[] )
    try:
        config = {"recursion_limit": 15}
        final_state = debris_agent_graph.invoke(initial_state, config=config)
        logging.info("Debris Analysis Agent workflow finished.")
        if final_state.get('error_message') and not final_state.get('final_report'):
             return f"Agent finished with error: {final_state['error_message']}"
        return final_state.get("final_report", "Error: Final report not found in agent state.")
    except Exception as e:
        logging.error(f"Exception during agent graph invocation: {e}", exc_info=True)
        return f"Critical Error during agent execution: {e}"


# --- Direct Execution Example ---
# ... (Direct execution block remains the same) ...
if __name__ == "__main__":
    print("Running Debris Agent Standalone Test...")
    report = run_debris_analysis_agent()
    print("\n--- FINAL REPORT ---")
    print(report)

