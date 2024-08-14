# FarmBuddy

## Overview

<img src="assets/home_page.png" class="img-responsive" alt="">

> A web application that uses AI to analyze crops for pests and diseases then advises framers on what to do.

 The application consists of

## Application Demo

The application supports the following operations:

1. Account registration using your Google account.
2. Analyzing images for either pests or diseases
3. Chatting with an expert.
4. Finding aggrovets

<p align=center>
  <img src="assets/farm-buddy.gif" />
</p>

## Local Setup

To work with the application locally, first make sure the following are present:

1. You have a groq API key
2. A Google Maps API Key.

Folow these steps to start the application:

1. Clone the project repo:

```sh
  git clone https://github.com/twyle/FarmBuddy.git
```

2. Navigate to the project directory, then create the project secrets ``(.env file)``:
```sh
cd FarmBuddy/farm-assistant/app
# Linux
touch .env
# Windows
# Create a file called .env in FarmBuddy/farm-assistant/app
```

3. Add the project secrets in the ``(.env file)``. This is how it should look:

```sh
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_MAPS_API_KEY=AIxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Replace /home/lyle/Downloads/test.pt with the full path to the model
MAIZE_MODEL_DIRECTORY=/home/lyle/Downloads/test.pt
PEST_MODEL_DIRECTORY=/home/lyle/Downloads/PestNet.pkl
```
3. Install the project requrements
```sh
  pip install -r requirements.txt
```
4. Run the application
```sh
  python manage.py
```
5. View the running application, click on ths 
[link (http://localhost:8000)](http://localhost:8000)