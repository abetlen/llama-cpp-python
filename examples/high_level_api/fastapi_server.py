"""Example FastAPI server for llama.cpp.

To run this example:

```bash
pip install fastapi uvicorn sse-starlette
export MODEL=../models/7B/...
```

Then run:
```
uvicorn --factory llama_cpp.server.app:create_app --reload
```

or

```
python3 -m llama_cpp.server
```

Then visit http://localhost:8000/docs to see the interactive API docs.


To actually see the implementation of the server, see llama_cpp/server/app.py

"""
import os
import uvicorn
from llama_cpp.server.app import create_app
import asyncio

if __name__ == "__main__":
    app = create_app()

    try:
        # Run the server with timeout config (graceful shutdown handling)
        uvicorn.run(
            app,
            host=os.getenv("HOST", "localhost"),
            port=int(os.getenv("PORT", 8000)),
            timeout_keep_alive=10,  # Optional: disconnect inactive clients after 10s
            timeout_notify=5        # Optional: timeout for graceful shutdown notification
        )
    except asyncio.TimeoutError:
        print("⏰ Server startup timed out.")
    except Exception as e:
        print(f"🚨 An unexpected error occurred: {e}")
