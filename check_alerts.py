from database import SessionLocal
from models import Alert, Inventory
from sqlalchemy.orm import joinedload

db = SessionLocal()
try:
    # Get alerts with material information
    alerts = db.query(Alert).options(joinedload(Alert.material)).filter(Alert.is_active == True).all()
    print('Current active alerts:')
    for alert in alerts:
        print(f'Alert ID: {alert.id}')
        print(f'Type: {alert.alert_type}')
        print(f'Message: {alert.message}')
        material_name = alert.material.material_name if alert.material else 'Unknown'
        print(f'Material: {material_name}')
        print(f'Created: {alert.date_created}')
        print('---')
finally:
    db.close()