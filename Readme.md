<p align="center">
  <h1 align="center">BuildOptima: AI-Powered Construction Site Management</h1>
  <p align="center">
    Streamlining construction management with intelligent insights.
  </p>
  </p>

## Overview

BuildOptima is a web application designed to streamline construction site management tasks. It provides core functionalities for tracking materials, suppliers, costs, consumption, and waste, enhanced with AI-driven analysis and suggestions using agentic AI workflows built with LangGraph.

The goal is to optimize resource allocation, minimize waste, improve procurement decisions, and provide actionable insights for better site management.

## Features ✨

**Core Management Modules:**

* **Materials Management:** Track inventory levels, units, reorder points, and associated suppliers.
* **Supplier Management:** Maintain a database of suppliers with contact details, lead times, and reliability ratings.
* **Consumption Tracking:** Log material usage against specific projects.
* **Cost Tracking:** Record material purchase costs.
* **Waste/Debris Management:** Log waste generated, including material type, quantity, reason, and project.

**AI-Powered Features:** 🧠

* **Debris Analysis Agent:**
    * Analyzes logged waste data to identify trends (top materials, reasons, projects).
    * (Placeholder) Researches local disposal and recycling options.
    * Generates actionable strategies to reduce future waste.
    * Presents findings in a structured report within the UI.
* **Inventory Optimization Agent:**
    * Analyzes consumption history and current inventory levels.
    * Estimates material demand trends.
    * Uses DuckDuckGo Search for basic price context research (for top consumed items).
    * Calculates estimated safety stock based on supplier lead times and reliability.
    * Suggests optimized reorder points and order quantities.
    * Assesses potential stockout or overstock risks.
    * Presents findings in a structured report within the UI.

## Technology Stack 🛠️

* **Backend:** Python, FastAPI
* **Database:** SQLite (via SQLAlchemy ORM)
* **Data Validation:** Pydantic
* **AI Orchestration:** LangGraph, Langchain
* **LLM Integration:** OpenRouter (compatible with various models like Mistral, Gemini, etc.)
* **Web Search Tool:** DuckDuckGo Search (`duckduckgo-search`, `langchain_community`)
* **Frontend Templating:** Jinja2
* **Frontend Styling:** Tailwind CSS (via CDN)
* **Frontend Interaction:** Vanilla JavaScript, Marked.js (for Markdown rendering)

## Project Structure 📁

<details>
<summary>Click to view project structure</summary>

material_management/├── agents/│   ├── init.py│   ├── debris_agent.py       # Logic for the waste analysis agent│   └── inventory_agent.py    # Logic for the inventory analysis agent├── static/│   └── styles.css            # Custom CSS (if any)├── templates/│   ├── base.html             # Base HTML template│   ├── consumption.html      # Consumption page│   ├── costs.html            # Costs page│   ├── index.html            # Home/Dashboard page│   ├── materials.html        # Materials/Inventory page + AI report display│   ├── suppliers.html        # Suppliers page│   └── waste.html            # Waste page + AI report display├── .env                      # Environment variables (API keys, etc. - NOT COMMITTED)├── crud.py                   # Database Create, Read, Update, Delete operations├── database.py               # SQLAlchemy setup and database session management├── main.py                   # FastAPI application, routes, API endpoints├── models.py                 # SQLAlchemy database models├── requirements.txt          # Python package dependencies├── schemas.py                # Pydantic data models for validation and serialization└── construction_materials.db # SQLite database file (created on first run)
</details>

## Setup and Installation ⚙️

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd material_management
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create `.env` File:**
    Create a file named `.env` in the `material_management` root directory. Add your OpenRouter API key:
    ```dotenv
    OPENROUTER_API_KEY="your_openrouter_api_key_here"
    # Optional: Specify a default model if desired
    # OPENROUTER_MODEL_NAME="mistralai/mistral-7b-instruct"
    ```
    *(**Note:** Ensure the `.env` file is listed in your `.gitignore` file to avoid committing secrets.)*

## Running the Application ▶️

1.  **Start the FastAPI Server:**
    From the `material_management` directory (where `main.py` is located):
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    * `--reload`: Automatically restarts the server when code changes are detected.
    * `--host 0.0.0.0`: Makes the server accessible on your local network.
    * `--port 8000`: Specifies the port number.

2.  **Access the Application:**
    Open your web browser and navigate to `http://localhost:8000` or `http://<your-local-ip>:8000`.

## Key API Endpoints 🔌

* `/` : Home page / Dashboard
* `/materials` : Materials inventory page
* `/suppliers` : Suppliers list page
* `/consumption` : Consumption log page
* `/costs` : Costs log page
* `/waste` : Waste log page
* `/api/materials`: CRUD operations for materials
* `/api/suppliers`: CRUD operations for suppliers
* `/api/consumption`: CRUD operations for consumption records
* `/api/costs`: CRUD operations for cost records
* `/api/waste`: CRUD operations for waste records
* `/api/debris/report`: (GET) Triggers the Debris Analysis Agent and returns the report.
* `/api/inventory/report`: (GET) Triggers the Inventory Analysis Agent and returns the report.

## Future Enhancements 🚀

* Implement real-time web search results in agent reports.
* Refine UI/UX for report display (e.g., interactive charts, better visual hierarchy).
* Add user authentication and authorization.
* Develop more sophisticated forecasting models.
* Implement project management features more deeply.
* Integrate cost data into optimization suggestions more robustly.
* Refactor database access in tools to use FastAPI dependency injection.
* Add more comprehensive unit/integration tests.

