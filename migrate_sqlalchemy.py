from sqlalchemy import create_engine, text
from shutil import copy2
from datetime import datetime
import os
import sys

DB_URL = "sqlite:///./construction_materials.db"
BACKUP = f"construction_materials.db.bak.{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"

if not os.path.exists("construction_materials.db"):
    print("DB not found.")
    sys.exit(1)
copy2("construction_materials.db", BACKUP)
print("Backup:", BACKUP)

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
with engine.begin() as conn:
    res = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='inventory'")).fetchone()
    if not res:
        print("Table inventory not found.")
        sys.exit(1)
    # Add columns if not exists — sqlite ignores IF NOT EXISTS for ADD COLUMN, so we need to check PRAGMA
    cols = [r[1] for r in conn.execute(text("PRAGMA table_info('inventory')")).fetchall()]
    if 'estimated_quantity' not in cols:
        conn.execute(text("ALTER TABLE inventory ADD COLUMN estimated_quantity REAL"))
        print("Added estimated_quantity")
    else:
        print("estimated_quantity exists")
    if 'estimated_unit_price' not in cols:
        conn.execute(text("ALTER TABLE inventory ADD COLUMN estimated_unit_price REAL"))
        print("Added estimated_unit_price")
    else:
        print("estimated_unit_price exists")
    if 'estimated_total_value' not in cols:
        conn.execute(text("ALTER TABLE inventory ADD COLUMN estimated_total_value REAL"))
        print("Added estimated_total_value")
    else:
        print("estimated_total_value exists")
print("Done.")
