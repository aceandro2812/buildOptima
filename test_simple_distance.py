import requests

print('🧪 Testing Distance Calculator')
print('=' * 40)

try:
    response = requests.get('http://localhost:8000/api/gis/nearest-suppliers/1?limit=3')
    if response.status_code == 200:
        data = response.json()
        print('✅ Distance calculation API working!')
        project_name = data['project']['name']
        print(f'Project: {project_name}')
        suppliers = data['nearest_suppliers']
        print(f'Found {len(suppliers)} nearest suppliers:')
        
        for i, supplier in enumerate(suppliers):
            name = supplier['name']
            distance = supplier['distance_formatted']
            print(f'  {i+1}. {name}: {distance}')
    else:
        print(f'❌ API error: {response.status_code}')

except Exception as e:
    print(f'❌ Error: {e}')