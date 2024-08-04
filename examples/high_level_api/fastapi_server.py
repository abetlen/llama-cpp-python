"""Example FastAPI server for llama.cpp.

To run this example:

```bash
pip install fastapi hypercorn sse-starlette
export MODEL=../models/7B/...
```

Then run:
```
hypercorn --factory llama_cpp.server.app:create_app --reload
```

or

```
python3 -m llama_cpp.server
```

Then visit http://localhost:8000/docs to see the interactive API docs.


To actually see the implementation of the server, see llama_cpp/server/app.py

"""
import os
import hypercorn.asyncio
from hypercorn.config import Config

from llama_cpp.server.app import create_app

if __name__ == "__main__":
    app = create_app()

    config = Config()
    host = os.getenv('HOST', 'localhost')
    port = int(os.getenv('PORT', 8000))
    
    config.bind = [f"{host}:{port}", f"[::]:{port}"]
    
    hypercorn.asyncio.run(app, config)
