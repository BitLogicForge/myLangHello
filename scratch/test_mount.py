from fastapi import FastAPI
import sys
from pathlib import Path

# Add root directory to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

app = FastAPI()

try:
    from chainlit.utils import mount_chainlit
    mount_chainlit(app=app, target="chat_app.py", path="/chat")
    print("✅ mount_chainlit completed successfully!")
except Exception as e:
    import traceback
    print("❌ mount_chainlit failed with exception:")
    traceback.print_exc()
