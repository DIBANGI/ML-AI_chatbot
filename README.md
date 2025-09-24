AI Customer Service Chatbot (Google Colab Guide)
This guide provides step-by-step instructions to set up and run the entire full-stack chatbot application within a Google Colab environment.

## Prerequisites
A Google Account to use Google Drive and Colab.

A Groq API Key. You can get one from the Groq Console.

A free ngrok Account to create a public URL for the chatbot.

## Step 1: Upload Project to Google Drive
Make sure all your project files (including the static folder, .py files, .csv files, etc.) are in a single main folder on your local machine. Let's call it my_chatbot_project.

Open your Google Drive.

Drag and drop the entire my_chatbot_project folder into your Google Drive to upload it.

## Step 2: Set Up the Colab Notebook
Now, we will use a Colab notebook to run the code from your Google Drive.

### 1. Open and Mount Drive
Open a new Colab notebook and run the following code in the first cell to connect it to your Google Drive. A popup will appear asking for permission.

from google.colab import drive
drive.mount('/content/drive')

### 2. Navigate to Your Project Directory
Use the %cd command to move into your project folder. Make sure to replace my_chatbot_project with the actual name of your folder.

# IMPORTANT: Change "my_chatbot_project" to the name of your folder!
%cd /content/drive/MyDrive/my_chatbot_project/

## Step 3: Configuration and Installation
Run the following cells in order.

### 1. Create the Environment File
This cell creates the .env file inside your project folder and writes your Groq API key to it.

⚠️ Important: Paste your actual Groq API key where it says "gsk_YourActualGroqApiKeyHere".

# Paste your Groq API key below
groq_api_key = "gsk_YourActualGroqApiKeyHere"

with open('.env', 'w') as f:
    f.write(f'GROQ_API_KEY="{groq_api_key}"\n')
    f.write('DEBUG=True\n')
    f.write('PORT=8001\n')

print(".env file created successfully!")

### 2. Install Dependencies
This command installs all the necessary Python libraries from your requirements.txt file.

!pip install -r requirements.txt

## Step 4: Prepare the Data
These commands will build your databases.

### 1. Create the SQLite Database
This script reads your cleaned_inventory.csv and creates the vasuki_inventory.db file.

!python import_data.py

### 2. Populate the Vector Store
This script reads your text files and the SQLite database to create the vector embeddings for the RAG system.

!python data_loader_main.py

## Step 5: Run the Chatbot Application
This final cell uses pyngrok to create a public URL for your application and then starts the FastAPI server.

import nest_asyncio
from pyngrok import ngrok
import uvicorn
import os

# ⚠️ IMPORTANT: You must add your ngrok authtoken to run the server.
# 1. Sign up for a free account at https://dashboard.ngrok.com/signup
# 2. Copy your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
# 3. Paste your token between the quotes below.
NGROK_AUTH_TOKEN = "PASTE_YOUR_NGROK_AUTHTOKEN_HERE" 

if NGROK_AUTH_TOKEN == "PASTE_YOUR_NGROK_AUTHTOKEN_HERE":
  print("ERROR: Please paste your ngrok authtoken into the NGROK_AUTH_TOKEN variable.")
else:
  ngrok.set_auth_token(NGROK_AUTH_TOKEN)

  nest_asyncio.apply()

  # Start ngrok tunnel
  public_url = ngrok.connect(8001)
  print(f"✅ Your Chatbot is live at: {public_url}")

  # Run the Uvicorn server
  uvicorn.run("main:app", host="0.0.0.0", port=8001)

### Accessing Your Chatbot
After running the cell above, you will see an output line that looks like this:

✅ Your Chatbot is live at: NgrokTunnel: "https://<some-random-characters>.ngrok-free.app"

Copy the https://... URL and paste it into a new browser tab. Your chatbot interface will load, and you can start interacting with it!