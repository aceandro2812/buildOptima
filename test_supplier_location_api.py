import requests

print('🧪 Testing Supplier Location API')
print('=' * 30)

base_url = 'http://localhost:8000'

try:
    # Test creating a supplier with location
    new_supplier = {
        'name': 'Test Location Supplier',
        'contact_person': 'Jane Doe',
        'phone': '+91-9876543210',
        'email': 'jane@testlocation.com',
        'address': 'Test Location Area',
        'latitude': 19.1234,
        'longitude': 72.5678,
        'lead_time_days': 5,
        'reliability_rating': 4.0
    }
    
    response = requests.post(f'{base_url}/api/suppliers', json=new_supplier)
    if response.status_code == 201:
        created_supplier = response.json()
        supplier_id = created_supplier['id']
        print(f'✅ Supplier created with ID: {supplier_id}')
        lat = created_supplier.get('latitude')
        lng = created_supplier.get('longitude')
        print(f'📍 Location: {lat}, {lng}')
        
        # Test retrieving the supplier
        get_response = requests.get(f'{base_url}/api/suppliers/{supplier_id}')
        if get_response.status_code == 200:
            retrieved_supplier = get_response.json()
            ret_lat = retrieved_supplier.get('latitude')
            ret_lng = retrieved_supplier.get('longitude')
            print(f'✅ Retrieved supplier location: {ret_lat}, {ret_lng}')
        
        # Cleanup
        requests.delete(f'{base_url}/api/suppliers/{supplier_id}')
        print('🧹 Test supplier deleted')
        
    else:
        print(f'❌ Failed to create supplier: {response.status_code}')
        print(response.text)

except Exception as e:
    print(f'❌ Error: {e}')