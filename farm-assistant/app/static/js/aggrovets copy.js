// Initialize and add the map
let map;

async function getPosition(){

}

asyn function getAPIKey(){
    
}

async function initMap() {
  // The location of Juja
  const position = { lat: -1.0912, lng: 37.0117 };
  // Request needed libraries.
  //@ts-ignore
  const { Map } = await google.maps.importLibrary("maps");
  const { AdvancedMarkerElement } = await google.maps.importLibrary("marker");

  // The map, centered at Juja
  map = new Map(document.getElementById("map"), {
    zoom: 15,
    center: position,
    mapId: "DEMO_MAP_ID",
  });

  // The marker, positioned at Juja
  const juja = new AdvancedMarkerElement({
    map: map,
    position: position,
    title: "Juja",
  });

  const p = { lat: -1.1653991, lng: 37.0877019 };
  const p1 = { lat: -1.1055352, lng: 37.0145696 };
  const p2 = { lat: -1.1061188, lng: 37.014942 };
  const p3 = { lat: -1.1301227, lng: 37.0128704 };

  const m1 = new AdvancedMarkerElement({
    map: map,
    position: p,
    title: "Aggrovet",
  });
  const m2 = new AdvancedMarkerElement({
    map: map,
    position: p1,
    title: "Aggrovet",
  });
  const m3 = new AdvancedMarkerElement({
    map: map,
    position: p2,
    title: "Aggrovet",
  });
  const m4 = new AdvancedMarkerElement({
    map: map,
    position: p3,
    title: "Aggrovet",
  });
}

initMap();