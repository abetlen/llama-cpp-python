"""Example FastAPI server for llama.cpp.

To run this example:

```bash
python examples/high_level_api/fastapi_server.py --model <path/to/your/model>
```

To actually see the implementation of the server, see llama_cpp/server/app.py

"""

import os
import uvicorn
import argparse

from llama_cpp.server.app import create_app
from llama_cpp.server.cli import parse_model_from_args
from llama_cpp.server.settings import (
    ServerSettings,
    ModelSettings,
)

from fastapi import APIRouter

# you can even customize your own router above the llama_cpp server

test_router = APIRouter()

@test_router.get("/llm")
def test_llm():
    return {'status': 'OK'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Path to your model",
    )
    args = parser.parse_args()
    server_settings = parse_model_from_args(ServerSettings, args)
    model_settings = [parse_model_from_args(ModelSettings, args)]
    
    """
        By default, we can use:
            app = create_app()
        to create a FastAPI app for serving.
        
        If you add your custom routers, you can simply use:
            app = create_app(custom_routers=<your_router_here>)
        to register your custom routers above llama_cpp.server's routers.
    """
    
    app = create_app(
        server_settings=server_settings,
        model_settings=model_settings,
        custom_routers=test_router
    )

    uvicorn.run(
        app, host=os.getenv("HOST", "localhost"), port=int(os.getenv("PORT", 8000))
    )
