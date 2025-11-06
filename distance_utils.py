import math

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    
    c = 2 * math.asin(math.sqrt(a))
    earth_radius_km = 6371.0
    distance = earth_radius_km * c
    
    return distance

def format_distance(distance_km):
    if distance_km < 1.0:
        meters = distance_km * 1000
        return f"{meters:.0f}m"
    elif distance_km < 10.0:
        return f"{distance_km:.1f}km"
    else:
        return f"{distance_km:.0f}km"

def find_nearest_suppliers(project_lat, project_lon, suppliers, limit=5):
    suppliers_with_distance = []
    
    for supplier in suppliers:
        if supplier.latitude is not None and supplier.longitude is not None:
            distance = haversine_distance(
                project_lat, project_lon,
                supplier.latitude, supplier.longitude
            )
            
            supplier_data = {
                'id': supplier.id,
                'name': supplier.name,
                'contact_person': supplier.contact_person,
                'phone': supplier.phone,
                'address': supplier.address,
                'latitude': supplier.latitude,
                'longitude': supplier.longitude,
                'distance_km': distance,
                'distance_formatted': format_distance(distance)
            }
            suppliers_with_distance.append(supplier_data)
    
    suppliers_with_distance.sort(key=lambda x: x['distance_km'])
    return suppliers_with_distance[:limit]

def calculate_project_supplier_distances(project_lat, project_lon, suppliers):
    suppliers_with_distance = []
    
    for supplier in suppliers:
        if supplier.latitude is not None and supplier.longitude is not None:
            distance = haversine_distance(
                project_lat, project_lon,
                supplier.latitude, supplier.longitude
            )
            
            supplier_data = {
                'id': supplier.id,
                'name': supplier.name,
                'contact_person': supplier.contact_person,
                'phone': supplier.phone,
                'address': supplier.address,
                'latitude': supplier.latitude,
                'longitude': supplier.longitude,
                'distance_km': distance,
                'distance_formatted': format_distance(distance)
            }
            suppliers_with_distance.append(supplier_data)
    
    suppliers_with_distance.sort(key=lambda x: x['distance_km'])
    return suppliers_with_distance