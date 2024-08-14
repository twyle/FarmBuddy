import googlemaps
import os


def geocode_location(location: str) -> dict[str, float]:
    gmaps = googlemaps.Client(key=os.environ['GOOGLE_MAPS_API_KEY'])
    geocode_results: dict = gmaps.geocode(location)
    location: dict[str, float] = geocode_results[0]['geometry']['location']
    address: str = geocode_results[0]['formatted_address']
    return {'location': location, 'address': address}


def get_aggrovet_details(place_id: str) -> dict:
    gmaps = googlemaps.Client(key=os.environ['GOOGLE_MAPS_API_KEY'])
    result: dict = gmaps.place(place_id=place_id)
    aggrovet: dict = {}
    aggrovet['formatted_phone_number'] = result['result']['formatted_phone_number']
    return aggrovet

def get_agrovets(location: str) -> str:
    """Useful when you need to get agrovets in a given location. Give it a query, such as 
    agrovets in Nairobi, Kenya. Only use this tool to find aggrovets!
    """
    gmaps = googlemaps.Client(key=os.environ['GOOGLE_MAPS_API_KEY'])
    results = gmaps.places(query=f'Get me aggrovets in {location}')
    aggrovet_locations: list[str] = list()
    for result in results['results']:
        try:
            aggrovet: dict = dict()  
            aggrovet['formatted_address'] = result['formatted_address']
            aggrovet['name'] = result['name']
            aggrovet['location'] = result['geometry']['location']
            aggrovet['place_id'] = result['place_id']
            aggrovet['details'] = get_aggrovet_details(place_id=result['place_id'])
            aggrovet_locations.append(aggrovet)
        except Exception as e:
            print(e)
    return aggrovet_locations