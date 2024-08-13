import googlemaps
import os


def geocode_location(location: str) -> dict[str, float]:
    gmaps = googlemaps.Client(key=os.environ['GOOGLE_MAPS_API_KEY'])
    geocode_results: dict = gmaps.geocode(location)
    location: dict[str, float] = geocode_results[0]['geometry']['location']
    address: str = geocode_results[0]['formatted_address']
    return {'location': location, 'address': address}


def get_agrovets(location: str) -> list[dict]:
    """Useful when you need to get agrovets in a given location. Give it a query, such as 
    agrovets in Nairobi, Kenya. Only use this tool to find aggrovets!
    """
    gmaps = googlemaps.Client(key=os.environ['GOOGLE_MAPS_API_KEY'])
    results = gmaps.places(query=f'Get me aggrovets in {location}')
    aggrovet_locations: list[str] = list()
    for result in results['results']:
        try:
            bussiness: dict = dict()  
            bussiness['formatted_address'] = result['formatted_address']
            bussiness['name'] = result['name']
            bussiness['location'] = result['geometry']['location']
            aggrovet_locations.append(bussiness)
        except:
            pass
    return aggrovet_locations