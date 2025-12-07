import json
import requests
from pathlib import Path
import time 
import os
import httpx
from llama_api_client import LlamaAPIClient, DefaultHttpxClient

with open("mckinney-api.key", "r") as f:
    API_KEY = f.read().strip()

client = LlamaAPIClient(
    # Or use the `LLAMA_API_CLIENT_BASE_URL` env var
    api_key=API_KEY,
    base_url="https://genai.rcac.purdue.edu/api",
    http_client=DefaultHttpxClient(),
)


response = client.chat.completions.create(
    model="llama4:latest",
    messages = [
        {
            "role": "user",
            "content": "Pick an example animal from the list: dog, cat, monkey, bird."
        },
        {
            "role": "user",
            "content": "How many legs does that animal have?"
        }
]
)

#print(response.completion_message.content.text)
print(response)