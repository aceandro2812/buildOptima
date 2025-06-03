#!/bin/bash
# run.sh

# Navigate to script location (useful if run from elsewhere)
cd "$(dirname "$0")"

# Activate virtual environment
source venv/Scripts/activate

# Run FastAPI app
uvicorn main:app --reload
