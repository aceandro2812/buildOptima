from main import app
from fastapi.testclient import TestClient

client = TestClient(app)
response = client.get('/api/debris/report')
result = response.json()
report = result['report']

sections = ['Executive Summary', 'Waste Analysis', 'Disposal Options', 'Reduction Strategies']
print('=== REPORT SECTIONS CHECK ===')
for section in sections:
    status = "✅ FOUND" if section in report else "❌ MISSING"
    print(f'{section}: {status}')

print(f'\n=== REPORT STATS ===')
print(f'Total length: {len(report)} characters')
print(f'Contains timestamp: {"✅ YES" if "Report Generated:" in report else "❌ NO"}')
print(f'Contains location: {"✅ YES" if "Thane" in report else "❌ NO"}')