# agents/debris_agent.py (Absolute Imports, Pydantic V2, Local Web Search)

import os
import json
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv
from typing import List, TypedDict, Sequence, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel # Use Pydantic V2 directly
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from tools.local_search import LocalSearchTool # <-- Import Local Search Tool

# --- Add parent directory to path for standalone execution ---
if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# --- Database Access (Using Absolute Imports) ---
from database import SessionLocal
# Use the CRUD function which loads relationships
from crud import get_waste_data as crud_get_waste_data

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Setup ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "mistralai/mistral-7b-instruct")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found")

# --- LLM Configuration ---
try:
    llm = ChatOpenAI( model=OPENROUTER_MODEL_NAME, openai_api_key=OPENROUTER_API_KEY, openai_api_base="https://openrouter.ai/api/v1", temperature=0.7 )
    logging.info(f"LLM initialized with model: {OPENROUTER_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}")
    raise

# --- Tool Definitions ---
# Internal Pydantic model for the agent's tool
class WasteRecord(BaseModel):
    id: int
    material_name: str
    project_name: str # This stores the project *name* string
    quantity_wasted: float
    unit: str
    reason: Optional[str] = None
    preventive_measures: Optional[str] = None
    date_recorded: Optional[str] = None

# --- Tool with FIX ---
@tool
def get_waste_database_records() -> List[WasteRecord]:
    """ Fetches all waste records from the database. """
    db = SessionLocal()
    result_list = []
    skipped_count = 0
    try:
        # Use the CRUD function that eager loads relationships
        waste_records_orm = crud_get_waste_data(db)
        for record in waste_records_orm:
            # Basic checks
            if record.material_id is None or record.project_id is None:
                logging.warning(f"Skipping waste record ID {record.id} due to missing material_id or project_id")
                skipped_count += 1
                continue

            # Get material details safely
            material_info = {"material_name": "N/A", "unit": "N/A"}
            if record.material:
                material_info["material_name"] = record.material.material_name
                material_info["unit"] = record.material.unit

            # --- FIX: Access project name via project_rel ---
            project_name = record.project_rel.name if record.project_rel else "N/A"
            # --- END FIX ---

            try:
                # Create the Pydantic WasteRecord
                pydantic_record = WasteRecord(
                    id=record.id,
                    material_name=material_info["material_name"],
                    project_name=project_name, # Use the correctly fetched project name
                    quantity_wasted=record.quantity_wasted,
                    unit=material_info["unit"],
                    reason=record.reason,
                    preventive_measures=record.preventive_measures,
                    date_recorded=record.date_recorded.isoformat() if record.date_recorded else None
                )
                result_list.append(pydantic_record)
            except Exception as pydantic_error:
                 logging.warning(f"Skipping waste record ID {record.id} due to Pydantic validation error: {pydantic_error}")
                 skipped_count += 1

        if skipped_count > 0:
            logging.warning(f"Skipped {skipped_count} waste records due to missing IDs or validation errors.")
        logging.info(f"Successfully fetched {len(result_list)} waste records from DB.")
    except Exception as e:
        # Log the specific error that occurred during DB fetch or processing
        logging.error(f"Error fetching/processing waste data from DB: {e}", exc_info=True)
        # Optionally return empty list or re-raise depending on desired agent behavior
        return []
    finally:
        db.close()
    return result_list
# --- End Tool with FIX ---


# Initialize the Local Search tool instance
search_tool = LocalSearchTool(max_results=3) # <-- Initialize Local Search Tool

# --- Agent State Definition ---
class DebrisAnalysisState(TypedDict):
    raw_waste_data: List[WasteRecord] # Expecting list of agent's internal WasteRecord
    waste_summary: Optional[str]
    disposal_options: Optional[str]
    reduction_strategies: Optional[str]
    final_report: Optional[str]
    error_message: Optional[str]
    messages: Sequence[BaseMessage]

# --- Agent Node Functions ---
def fetch_data_node(state: DebrisAnalysisState) -> DebrisAnalysisState:
    logging.info("--- Node: Fetching Waste Data ---")
    try:
        # Call the fixed tool function
        raw_data: List[WasteRecord] = get_waste_database_records.invoke({})
        state['raw_waste_data'] = raw_data
        # Preview using model_dump for Pydantic v2
        data_preview = [record.model_dump(exclude={'id'}) for record in raw_data[:3]]
        state['messages'] = [HumanMessage( content=f"Raw waste data preview (first {len(data_preview)} records): {json.dumps(data_preview)}... Total records: {len(raw_data)}" )]
        logging.info(f"Fetched {len(raw_data)} records. Added preview to messages.")
        state['error_message'] = None
    except Exception as e:
        logging.error(f"Error in fetch_data_node: {e}", exc_info=True)
        state['error_message'] = f"Failed to fetch data: {e}"
        state['raw_waste_data'] = []
        state['messages'] = [SystemMessage(content=f"Error fetching waste data: {e}")]
    return state

def analyze_data_node(state: DebrisAnalysisState) -> DebrisAnalysisState:
    logging.info("--- Node: Analyzing Waste Data ---")
    if state.get('error_message'):
        logging.warning("Skipping analysis due to previous error.")
        return state
    raw_data = state.get('raw_waste_data', [])
    if not raw_data:
        summary = "No waste data found to analyze."
        logging.info(summary)
        state['waste_summary'] = summary
        if 'messages' not in state or not isinstance(state['messages'], list):
            state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=summary)]
        return state

    # Prepare a summary of the fetched data for the LLM prompt
    # Use the agent's internal WasteRecord structure
    data_summary_for_llm = []
    for r in raw_data[:10]:  # Limit preview size
        try:
            # Safely handle potentially None values
            material_name = r.material_name or "Unknown Material"
            quantity_wasted = r.quantity_wasted or 0
            unit = r.unit or "units"
            project_name = r.project_name or "Unknown Project"
            reason = (r.reason or "No reason provided")[:50]
            
            summary_item = f"Material '{material_name}' ({quantity_wasted} {unit}) from Project '{project_name}', Reason: {reason}..."
            data_summary_for_llm.append(summary_item)
        except Exception as e:
            logging.warning(f"Error processing waste record: {e}")
            continue
    
    # Ensure we have some data to analyze
    if not data_summary_for_llm:
        logging.warning("No valid waste data found for analysis")
        state['waste_summary'] = "No valid waste data available for analysis."
        state['error_message'] = "No valid waste data found"
        return state
    
    prompt = f"""You are a construction waste analyst for a site in Thane, India.
Based on the provided data preview and total record count, provide a concise summary highlighting:
- The most common types of materials wasted (if apparent from preview)
- The total number of waste records fetched ({len(raw_data)})
- Any recurring reasons for waste (if apparent from preview)
- Mention project names if they appear frequently in the preview

Data Preview (up to 10 records):
{chr(10).join(data_summary_for_llm)}

Generate a brief summary paragraph about the waste patterns observed."""
    # Validate prompt content before sending to LLM
    if not prompt or len(prompt.strip()) < 10:
        logging.error(f"Prompt is too short or empty: '{prompt[:100]}...'")
        state['error_message'] = "Generated prompt is empty or too short"
        state['waste_summary'] = "Error: Unable to generate analysis prompt."
        return state
    
    logging.info(f"Sending prompt to LLM (length: {len(prompt)} chars)")
    logging.info(f"Prompt preview: {prompt[:200]}...")
    
    # Try to clean the prompt content for Minimax
    cleaned_prompt = prompt.strip().replace('\n\n\n', '\n\n').replace('\t', ' ')
    if not cleaned_prompt:
        logging.error("Cleaned prompt is empty")
        state['error_message'] = "Cleaned prompt is empty"
        state['waste_summary'] = "Error: Prompt cleaning resulted in empty content."
        return state
    
    # Use HumanMessage for Minimax compatibility
    messages = [HumanMessage(content=cleaned_prompt)]
    
    try:
        response = llm.invoke(messages)
        if not response or not response.content:
            logging.error("LLM returned empty response")
            state['error_message'] = "LLM returned empty response"
            state['waste_summary'] = "Error: LLM analysis failed to generate content."
        else:
            summary = response.content.strip()
            state['waste_summary'] = summary
            logging.info(f"Waste data summary generated successfully (length: {len(summary)} chars)")
        
        if 'messages' not in state or not isinstance(state['messages'], list):
            state['messages'] = []
        state['messages'] = state['messages'] + [response] if response else []
        
    except Exception as e:
        logging.error(f"LLM invocation failed during analysis: {e}", exc_info=True)
        state['error_message'] = f"LLM error during analysis: {str(e)}"
        state['waste_summary'] = "Error during analysis."
        if 'messages' not in state or not isinstance(state['messages'], list):
            state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during analysis: {str(e)}")]
    return state

# --- Updated disposal_research_node ---
def disposal_research_node(state: DebrisAnalysisState) -> DebrisAnalysisState:
    """ Formulates a search query and executes it using DuckDuckGo to find disposal options. """
    logging.info("--- Node: Researching Disposal Options ---")
    if state.get('error_message') or "No waste data" in state.get('waste_summary', ""):
        logging.warning("Skipping disposal research due to previous error or no data.")
        state['disposal_options'] = "Disposal research skipped."
        if 'messages' not in state or not isinstance(state['messages'], list):
            state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=state['disposal_options'])]
        return state

    raw_data = state.get('raw_waste_data', [])
    location = "Thane, Maharashtra, India"
    # Get waste types from the agent's WasteRecord, ensuring uniqueness and filtering N/A
    waste_types = list(set(item.material_name for item in raw_data if item.material_name and item.material_name != "N/A"))

    if not waste_types:
        options = "Could not determine specific waste types from data. Skipping disposal research."
        logging.warning(options)
        state['disposal_options'] = options
        if 'messages' not in state or not isinstance(state['messages'], list):
            state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=options)]
        return state

    # Formulate query using LLM based on summary and waste types
    # Safely get waste summary and ensure it's not empty
    waste_summary = state.get('waste_summary', '').strip()
    if not waste_summary:
        waste_summary = "Construction waste analysis data available"
    
    # Ensure we have waste types to work with
    waste_types_str = ', '.join(waste_types[:4]) if waste_types else "construction materials"
    
    # Refined prompt for query formulation
    query_formulation_prompt = f"""Based on the waste summary and identified waste types below, formulate a concise web search query to find recycling facilities, disposal sites, and eco-friendly options for construction waste in {location}.

Identified waste types: {waste_types_str} (Total types: {len(waste_types)})
Waste Summary: {waste_summary}

Generate ONLY the search query string. Example: 'concrete debris recycling Thane cost'"""
    # Validate prompt content before sending to LLM
    if not query_formulation_prompt or len(query_formulation_prompt.strip()) < 10:
        logging.error(f"Query formulation prompt is too short or empty: '{query_formulation_prompt[:100]}...'")
        state['error_message'] = "Generated query formulation prompt is empty or too short"
        state['disposal_options'] = "Error: Unable to generate search query prompt."
        return state
    
    logging.info(f"Sending query formulation prompt to LLM (length: {len(query_formulation_prompt)} chars)")
    messages = [HumanMessage(content=query_formulation_prompt)]

    formulated_query = None
    try:
        response = llm.invoke(messages)
        if not response or not response.content:
            logging.error("LLM returned empty response for query formulation")
            state['error_message'] = "LLM returned empty response for query formulation"
            state['disposal_options'] = "Error: LLM failed to generate search query."
            return state
        
        # Clean up LLM output to get just the query string
        formulated_query = response.content.strip().strip('"').strip("'").strip()
        logging.info(f"LLM formulated search query: {formulated_query}")
        
        if 'messages' not in state or not isinstance(state['messages'], list):
            state['messages'] = []
        state['messages'] = state['messages'] + [response] # Log LLM response

    except Exception as e:
        logging.error(f"LLM invocation failed during disposal query formulation: {e}", exc_info=True)
        state['error_message'] = f"LLM error during disposal research prep: {str(e)}"
        state['disposal_options'] = "Error: Could not formulate search query."
        if 'messages' not in state or not isinstance(state['messages'], list):
            state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during disposal research prep: {str(e)}")]
        return state # Stop if query formulation fails

    # *** Perform Local Web Search ***
    if formulated_query:
        logging.info(f"Performing local web search for: {formulated_query}")
        try:
            # Use the local search tool for disposal options
            search_results = search_tool.multi_engine_search(formulated_query)
            
            # Format results for disposal context
            if search_results:
                disposal_info = f"Search results for '{formulated_query}':\n\n"
                for i, result in enumerate(search_results, 1):
                    disposal_info += f"{i}. {result.title}\n"
                    disposal_info += f"   URL: {result.url}\n"
                    disposal_info += f"   Info: {result.snippet}\n\n"
            else:
                disposal_info = f"No search results found for '{formulated_query}'"
            
            state['disposal_options'] = disposal_info
            state['messages'] = state['messages'] + [HumanMessage(content=f"Local Web Search Results:\n{disposal_info}")]
            logging.info(f"Local web search completed for query: {formulated_query}")
        except Exception as e:
            logging.error(f"Local web search failed: {e}", exc_info=True)
            error_msg = f"Error during local web search: {e}"
            state['disposal_options'] = f"Failed to perform search for query '{formulated_query}'. Error: {e}"
            # Decide if this is a critical error for the state
            # state['error_message'] = error_msg # Uncomment if search failure should halt the process
            state['messages'] = state['messages'] + [SystemMessage(content=error_msg)]
    else:
        # Handle case where LLM failed to generate a query
        state['disposal_options'] = "Search skipped as query could not be formulated."
        logging.warning("Search skipped as query could not be formulated.")
        state['messages'] = state['messages'] + [SystemMessage(content=state['disposal_options'])]

    return state
# --- End Updated disposal_research_node ---


def reduction_strategy_node(state: DebrisAnalysisState) -> DebrisAnalysisState:
    logging.info("--- Node: Generating Reduction Strategies ---")
    if state.get('error_message') or "No waste data" in state.get('waste_summary', ""):
        logging.warning("Skipping reduction strategies...")
        state['reduction_strategies'] = "Reduction strategies skipped due to prior errors or lack of data."
        if 'messages' not in state or not isinstance(state['messages'], list):
            state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=state['reduction_strategies'])]
        return state

    # Include disposal options context if available
    disposal_context = state.get('disposal_options', 'No disposal research performed.')
    waste_summary = state.get('waste_summary', 'No waste summary available.')
    # Get waste types again for context
    raw_data = state.get('raw_waste_data', [])
    waste_types = list(set(item.material_name for item in raw_data if item.material_name and item.material_name != "N/A"))
    waste_types_str = ', '.join(waste_types[:5]) + ('...' if len(waste_types) > 5 else '')

    # Ensure we have valid content for the prompt
    if not waste_summary or waste_summary.strip() == "No waste summary available.":
        waste_summary = "Construction waste data analysis completed"
    
    if not waste_types_str:
        waste_types_str = "construction materials"
    
    if not disposal_context or disposal_context.strip() == "No disposal research performed.":
        disposal_context = "General disposal and recycling options for construction waste"
    
    prompt = f"""You are a construction efficiency expert specializing in waste reduction in Thane, India.
Based on the waste summary, identified waste types, and disposal options, suggest 3-5 practical strategies to reduce construction waste.

Waste Summary: {waste_summary}

Identified Waste Types: {waste_types_str}

Disposal Context: {disposal_context[:500]}

Generate concise, bulleted recommendations for waste reduction."""
    
    # Validate prompt content before sending to LLM
    if not prompt or len(prompt.strip()) < 10:
        logging.error(f"Reduction strategies prompt is too short or empty: '{prompt[:100]}...'")
        state['error_message'] = "Generated reduction strategies prompt is empty or too short"
        state['reduction_strategies'] = "Error: Unable to generate reduction strategies prompt."
        return state
    
    logging.info(f"Sending reduction strategies prompt to LLM (length: {len(prompt)} chars)")
    messages = [HumanMessage(content=prompt)]
    
    try:
        response = llm.invoke(messages)
        if not response or not response.content:
            logging.error("LLM returned empty response for reduction strategies")
            state['error_message'] = "LLM returned empty response for reduction strategies"
            state['reduction_strategies'] = "Error: LLM failed to generate reduction strategies."
        else:
            strategies = response.content.strip()
            state['reduction_strategies'] = strategies
            logging.info(f"Reduction strategies generated successfully (length: {len(strategies)} chars)")
        
        if 'messages' not in state or not isinstance(state['messages'], list):
            state['messages'] = []
        state['messages'] = state['messages'] + [response] if response else []
        
    except Exception as e:
        logging.error(f"LLM invocation failed during strategy generation: {e}", exc_info=True)
        state['error_message'] = f"LLM error during strategy generation: {str(e)}"
        state['reduction_strategies'] = "Error during strategy generation."
        if 'messages' not in state or not isinstance(state['messages'], list):
            state['messages'] = []
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during strategy generation: {str(e)}")]
    return state

def compile_report_node(state: DebrisAnalysisState) -> DebrisAnalysisState:
    logging.info("--- Node: Compiling Report ---")
    if 'messages' not in state or not isinstance(state['messages'], list):
        state['messages'] = []
    if state.get('error_message') and not state.get('final_report'):
       state['final_report'] = f"## Report Error\n\nReport generation incomplete due to an error:\n\n* **Error:** {state['error_message']}"
       logging.warning("Compiling report based on error state.")
       state['messages'] = state['messages'] + [SystemMessage(content="Final report compiled with errors.")]
       return state

    # Prepare data snippets for context
    raw_data_summary = json.dumps(
        [r.model_dump(exclude={'id'}) for r in state.get('raw_waste_data', [])[:3]], indent=2
    )

    # Safely get all the content for the report
    waste_summary = state.get('waste_summary', '').strip()
    if not waste_summary:
        waste_summary = "Analysis could not be performed."
    
    disposal_options = state.get('disposal_options', '').strip()
    if not disposal_options:
        disposal_options = "Research could not be performed or is pending."
    
    reduction_strategies = state.get('reduction_strategies', '').strip()
    if not reduction_strategies:
        reduction_strategies = "Suggestions could not be generated."
    
    # Ensure raw_data_summary is not empty
    if not raw_data_summary or raw_data_summary.strip() == "[]":
        raw_data_summary = "No raw data available"
    
    summary_prompt = f"""Compile a professional waste analysis report for a construction site manager in Thane, India.

Structure: Use these Markdown H2 headings: ## Executive Summary, ## Waste Analysis, ## Disposal Options, ## Reduction Strategies

Tone: Clear, concise, professional language. Avoid jargon.

Content: Synthesize the provided analysis information under the correct headings.

Available Information:
Waste Summary: {waste_summary}

Disposal Options: {disposal_options}

Reduction Strategies: {reduction_strategies}

Generate the final report following these instructions. Add timestamp and location context."""
    # Validate prompt content before sending to LLM
    if not summary_prompt or len(summary_prompt.strip()) < 10:
        logging.error(f"Report compilation prompt is too short or empty: '{summary_prompt[:100]}...'")
        state['error_message'] = "Generated report compilation prompt is empty or too short"
        state['final_report'] = "## Report Error\n\nUnable to generate report compilation prompt."
        return state
    
    logging.info(f"Sending report compilation prompt to LLM (length: {len(summary_prompt)} chars)")
    messages = [HumanMessage(content=summary_prompt)]
    
    try:
        response = llm.invoke(messages)
        if not response or not response.content:
            logging.error("LLM returned empty response for report compilation")
            state['error_message'] = "LLM returned empty response for report compilation"
            state['final_report'] = "## Report Error\n\nLLM failed to generate report content."
        else:
            report_content = response.content.strip()
            timestamp = f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nLocation Context: Thane, Maharashtra, India\n\n"
            
            if not report_content.startswith("Report Generated:"):
                report_content = timestamp + report_content
            
            if "Note:" not in report_content[-200:]:
                report_content += "\n\n---\n*Note: This report is based on available data and AI analysis. Disposal options require verification based on search results.*"
            
            state['final_report'] = report_content
            logging.info(f"Report compiled successfully (length: {len(report_content)} chars)")
        
        state['messages'] = state['messages'] + [response] if response else []
        state['messages'] = state['messages'] + [SystemMessage(content="Final report compiled successfully.")]
        
    except Exception as e:
        logging.error(f"LLM invocation failed during report compilation: {e}", exc_info=True)
        state['error_message'] = f"LLM error during report compilation: {str(e)}"
        state['final_report'] = f"## Report Error\n\nFailed to compile report using LLM due to an error:\n\n* **Error:** {str(e)}"
        state['messages'] = state['messages'] + [SystemMessage(content=f"Error during report compilation: {str(e)}")]
    return state


# --- Build the Graph ---
logging.info("Building the Debris Agent workflow...")
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
    logging.info("Debris Agent workflow compiled successfully.")
except Exception as e:
    logging.error(f"Failed to compile Debris Agent workflow: {e}")
    raise


# --- Function to Invoke the Agent ---
def run_debris_analysis_agent() -> str:
    logging.info("Starting Debris Analysis Agent workflow...")
    # Initialize state correctly according to the TypedDict definition
    initial_state = DebrisAnalysisState(
        raw_waste_data=[],
        waste_summary=None,
        disposal_options=None,
        reduction_strategies=None,
        final_report=None,
        error_message=None,
        messages=[]
    )
    try:
        config = {"recursion_limit": 15}
        final_state = debris_agent_graph.invoke(initial_state, config=config)
        logging.info("Debris Analysis Agent workflow finished.")
        # Check for error message first if report compilation failed
        if final_state.get('error_message') and not final_state.get('final_report'):
             # Prefer final report even if error occurred earlier, unless report is missing
             if not final_state.get('final_report'):
                return f"Agent finished with error: {final_state['error_message']}"
        # Return final report or a default error message
        return final_state.get("final_report", "Error: Final report not found in agent state.")
    except Exception as e:
        logging.error(f"Exception during agent graph invocation: {e}", exc_info=True)
        return f"Critical Error during agent execution: {str(e)}"


# --- Direct Execution Example ---
if __name__ == "__main__":
    print("Running Debris Agent Standalone Test...")
    # Make sure the database is populated with some data first for a meaningful test
    print("Ensure database (construction_materials.db) has relevant data.")
    report = run_debris_analysis_agent()
    print("\n--- FINAL REPORT ---")
    print(report)
