# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from fastapi import FastAPI
from fastapi.testclient import TestClient
from chainlit.utils import mount_chainlit
import os

app = FastAPI()

try:
    target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../chat_app.py")
    mount_chainlit(app=app, target=target_path, path="/chat")
    print("✅ mount_chainlit called successfully")
except Exception as e:
    print(f"❌ mount_chainlit error: {e}")

client = TestClient(app)

print("\nTesting GET /chat")
try:
    response = client.get("/chat")
    print("Status code:", response.status_code)
    print("Headers:", dict(response.headers))
except Exception as e:
    print("Error:", e)

print("\nTesting GET /chat/")
try:
    response = client.get("/chat/")
    print("Status code:", response.status_code)
    print("Headers:", dict(response.headers))
    print("Content length:", len(response.text))
except Exception as e:
    print("Error:", e)
