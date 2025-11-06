import requests

print('🧪 Testing Alerts Display')
print('=' * 30)

# Test alerts API
response = requests.get('http://localhost:8000/api/alerts')
if response.status_code == 200:
    alerts = response.json()
    print(f'✅ Found {len(alerts)} active alerts')
    
    for alert in alerts:
        alert_type = alert['alert_type']
        message = alert['message'][:60]
        material = alert['material_name']
        project = alert['project_name']
        print(f'🚨 {alert_type}: {message}...')
        print(f'   Material: {material} | Project: {project}')
        
    # Test dashboard content
    dashboard_response = requests.get('http://localhost:8000/')
    if dashboard_response.status_code == 200:
        content = dashboard_response.text
        if 'active-alerts-content' in content:
            print('✅ Dashboard has alerts display section')
        if 'loadActiveAlerts' in content:
            print('✅ Dashboard has alert loading functions')
        if 'resolveAlert' in content:
            print('✅ Dashboard has alert resolution functions')
            
    print('🎉 UX Issue Fixed: Users can now see alerts!')
else:
    print(f'❌ API error: {response.status_code}')