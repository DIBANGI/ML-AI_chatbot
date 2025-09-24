import nest_asyncio
from pyngrok import ngrok
import uvicorn
import os
from dotenv import load_dotenv



# Set the ngrok auth token if you have one (optional, but recommended for stability)
# You can get a free token from your ngrok dashboard: https://dashboard.ngrok.com/get-started/your-authtoken
load_dotenv()
NGROK_AUTH_TOKEN = "31uG8HHGLf1oal5vESN0cMklH60_5iJmbUkVJZyGgGNw9Ho8m" # Paste your ngrok token here if you have one
if NGROK_AUTH_TOKEN:
  ngrok.set_auth_token(NGROK_AUTH_TOKEN)

nest_asyncio.apply()

# Start ngrok tunnel
public_url = ngrok.connect(8001)
print(f"✅ Your Chatbot is live at: {public_url}")

# Run the Uvicorn server
uvicorn.run("main:app", host="0.0.0.0", port=8001)
