const form = document.querySelector('#messageArea')
const submitButton = document.querySelector('#send')
const messageBox = document.querySelector('#text')
const messagesDiv = document.querySelector('#messageFormeight')


function appendUserMessage (userMessage) {
    let div = document.createElement('div')
    const date = new Date();
    const hour = date.getHours();
    const minute = date.getMinutes();
    const str_time = hour+":"+minute;
    var messageHtml = `
    <div class="d-flex justify-content-end mb-4">
        <div class="msg_cotainer_send">
            ${userMessage}
            <span class="msg_time_send">${str_time}</span>
        </div>
        <div class="img_cont_msg">
            <img src="https://i.ibb.co/d5b84Xw/Untitled-design.png" class="rounded-circle user_img_msg">
        </div>
    </div>
    `
    div.innerHTML = messageHtml
    messagesDiv.appendChild(div)
}


function appendBotMessage (botMessage) {
    let div = document.createElement('div')
    const date = new Date();
    const hour = date.getHours();
    const minute = date.getMinutes();
    const str_time = hour+":"+minute;
    var messageHtml = `
    <div class="d-flex justify-content-start mb-4">
        <div class="img_cont_msg">
            <img src="https://i.ibb.co/fSNP7Rz/icons8-chatgpt-512.png" class="rounded-circle user_img_msg">
        </div>
        <div class="msg_cotainer">
            ${ botMessage }
            <span class="msg_time">${ str_time }</span>
        </div>
    </div>
    `
    div.innerHTML = messageHtml
    messagesDiv.appendChild(div)
}


async function chatRequest (chatUrl = "", message = {}){
    const response = await fetch(chatUrl, {
      method: "POST", // *GET, POST, PUT, DELETE, etc.
      mode: "cors", // no-cors, *cors, same-origin
      cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
      credentials: "same-origin", // include, *same-origin, omit
      headers: {
        "Content-Type": "application/json",
      },
      redirect: "follow", // manual, *follow, error
      referrerPolicy: "no-referrer", // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
      body: JSON.stringify(message), // body data type must match "Content-Type" header
    });
    return response.json(); // parses JSON response into native JavaScript objects
  }

async function handleFormSubmission(event){
    event.preventDefault()
    const message = messageBox.value
    appendUserMessage(message)
    const url = '/chat'
    const chat = {role: 'user', message: message}
    const response = await chatRequest(url, chat)
    console.log(response)
    appendBotMessage(response['message'])
    messageBox.value = ""
}


form.addEventListener('submit', handleFormSubmission)