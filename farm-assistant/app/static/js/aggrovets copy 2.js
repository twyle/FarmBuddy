// Get the farmer postion
async function getPositionRequest(locationUrl){
    const response = await fetch(locationUrl, {
        method: "GET", // *GET, POST, PUT, DELETE, etc.
        mode: "cors", // no-cors, *cors, same-origin
        cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
        credentials: "same-origin", // include, *same-origin, omit
        headers: {
          "Content-Type": "application/json",
        },
        redirect: "follow", // manual, *follow, error
        referrerPolicy: "no-referrer", // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url, // body data type must match "Content-Type" header
      });
      return response.json();
}


async function getUser(){
    const url = "/analysis/location"
    const response = await getPositionRequest(url)
    console.log(response)
}

getUser()