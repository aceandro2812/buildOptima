# agents/procurement_agent.py
"""
Procurement Advisor Agent - Standardized with LangChain

- Orchestrates numeric procurement suggestions from crud.get_procurement_suggestions(...).
- For each suggestion, calls a standardized LLM interface (LangChain with OpenRouter)
  to produce a short, human-friendly explanation.
- Exposes run_procurement_advisor(project_id, lookback_days) which returns numeric
  suggestions annotated with an `ai_explanation`.

Environment:
- OPENROUTER_API_KEY: Required for LLM calls.
- OPENROUTER_MODEL_NAME: Optional, defaults to a Mistral model.

Notes:
- LLM is used only to *explain* numeric suggestions, not to compute them.
- Temperature is low (0.5) for consistent outputs.
- Follows the same structural and logging patterns as inventory_agent.py.
"""
import os
import json
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any

# --- LangChain/LLM Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# --- Add parent directory to path for standalone execution ---
if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# --- Database Access (Using Absolute Imports) ---
from database import SessionLocal
from crud import get_procurement_suggestions

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Setup ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "mistralai/mistral-7b-instruct")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

# --- LLM Configuration (Standardized) ---
try:
    llm = ChatOpenAI(
        model=OPENROUTER_MODEL_NAME,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.5
    )
    logging.info(f"LLM initialized successfully with model: {OPENROUTER_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}")
    raise

def call_llm_explanation(suggestion_summary: Dict[str, Any]) -> str:
    """
    Calls the standardized LLM to produce a clear procurement explanation for a given suggestion.
    If the LLM call fails, it provides a deterministic fallback explanation.

    Args:
        suggestion_summary: A dictionary with numeric fields like material_name,
                            current_quantity, recommended_order_qty, etc.
    """
    system_prompt = (
        "You are an expert procurement assistant. You will be given a small JSON describing a material's current "
        "stock, estimated demand, and recommended order quantity. Produce a short (2-3 sentences) human-friendly "
        "explanation and a one-line actionable recommendation for a site procurement manager. "
        "Be concise, avoid making up numbers not provided, and if data is missing, clearly state what is missing."
    )

    user_prompt = "Here is the data (JSON):\n" + json.dumps(suggestion_summary, indent=2, default=str)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    try:
        logging.info(f"Calling LLM for explanation on material: {suggestion_summary.get('material_name')}")
        response = llm.invoke(messages)
        explanation = response.content.strip()
        logging.info(f"LLM explanation received for {suggestion_summary.get('material_name')}")
        return explanation
    except Exception as e:
        logging.error(f"LLM invocation failed for procurement explanation: {e}", exc_info=True)
        # Fallback deterministic explanation
        try:
            material = suggestion_summary.get("material_name", "Material")
            rq = suggestion_summary.get("recommended_order_qty")
            rp = suggestion_summary.get("reorder_point")
            ss = suggestion_summary.get("safety_stock")
            cur = suggestion_summary.get("current_quantity")
            suppliers = suggestion_summary.get("top_suppliers", [])
            supplier_line = ""
            if suppliers:
                top = suppliers[0]
                supplier_line = f"Top supplier candidate: {top.get('supplier_name')} (lead {top.get('lead_time_days')}d, score {top.get('score')})."
            expl = (
                f"{material}: Current stock is {cur}. The recommended order quantity is {rq} to replenish stock, "
                f"considering a reorder point of {rp} and a safety stock of {ss}. "
                f"{supplier_line} If data like unit price or lead time is missing, please gather recent cost records or contact suppliers for quotes."
            )
            logging.warning(f"Using fallback explanation for {material}")
            return expl
        except Exception as fallback_e:
            logging.error(f"Fallback explanation generation failed: {fallback_e}")
            return "Procurement explanation is currently unavailable due to an unexpected error."


# --- Public entrypoint used by FastAPI ---
def run_procurement_advisor(project_id: int, lookback_days: int = 90):
    """
    Main function: fetches numeric suggestions, calls the standardized LLM to annotate
    each suggestion with an `ai_explanation`, and returns a result dictionary.

    Returns:
      A dictionary containing project info, generation timestamp, and a list of suggestions.
    """
    logging.info(f"Running procurement advisor for project_id: {project_id} with lookback: {lookback_days} days.")
    db = SessionLocal()
    try:
        numeric_results = get_procurement_suggestions(db, project_id, lookback_days=lookback_days)
        suggestions = numeric_results.get("suggestions", [])
        logging.info(f"Retrieved {len(suggestions)} numeric suggestions.")

        for s in suggestions:
            # Create a summary for the LLM prompt, keeping it concise
            summary = {
                "material_id": s.get("material_id"),
                "material_name": s.get("material_name"),
                "current_quantity": s.get("current_quantity"),
                "days_of_stock": s.get("days_of_stock"),
                "reorder_point": s.get("reorder_point"),
                "safety_stock": s.get("safety_stock"),
                "recommended_order_qty": s.get("recommended_order_qty"),
                "top_suppliers": s.get("supplier_scores", [])[:1]  # Only top 1 supplier for brevity
            }
            try:
                explanation = call_llm_explanation(summary)
            except Exception as e:
                # This is a safeguard; call_llm_explanation should handle its own errors
                logging.error(f"LLM explanation failed unexpectedly for material {s.get('material_name')}: {e}", exc_info=True)
                explanation = "AI explanation is currently unavailable."
            s["ai_explanation"] = explanation

        numeric_results["generated_at"] = datetime.utcnow().isoformat()
        logging.info("Procurement advisor run completed successfully.")
        return numeric_results
    except Exception as e:
        logging.critical(f"A critical error occurred in run_procurement_advisor: {e}", exc_info=True)
        # Return a structured error response
        return {
            "error": "Failed to generate procurement advice.",
            "details": str(e),
            "project_id": project_id,
            "generated_at": datetime.utcnow().isoformat()
        }
    finally:
        db.close()

# --- Direct Execution Example ---
if __name__ == "__main__":
    print("--- Running Procurement Advisor Standalone Test ---")
    # Ensure the database is populated for a meaningful test.
    # Example: Test with project_id=1
    project_id_to_test = 1
    print(f"Fetching procurement advice for project_id = {project_id_to_test}...")
    report = run_procurement_advisor(project_id=project_id_to_test)
    print("\n--- FINAL PROCUREMENT ADVICE ---")
    print(json.dumps(report, indent=2, default=str))
    print("--- Standalone Test Finished ---")