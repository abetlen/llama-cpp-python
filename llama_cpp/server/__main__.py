"""Example FastAPI server for llama.cpp.

To run this example:

```bash
pip install fastapi uvicorn sse-starlette pydantic-settings
export MODEL=../models/7B/...
```

Then run:
```
uvicorn llama_cpp.server.app:create_app --reload
```

or

```
python3 -m llama_cpp.server
```

Then visit http://localhost:8000/docs to see the interactive API docs.

"""
from __future__ import annotations

import os
import sys
import argparse

import uvicorn

from llama_cpp.server.app import create_app
from llama_cpp.server.settings import (
    Settings,
    ServerSettings,
    ModelSettings,
    read_config,
)
from llama_cpp.server.cli import add_args_from_model, parse_model_from_args


def main():
    description = "🦙 Llama.cpp python server. Host your own LLMs!🚀"
    parser = argparse.ArgumentParser(description=description)

    add_args_from_model(parser, Settings)
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to a config file to load.",
    )
    args = parser.parse_args()
    try:
        # Load server settings from config_file if provided
        config_file = os.environ.get("CONFIG_FILE", args.config_file)
        if config_file:
            server_settings, model_settings = read_config(config_file)
        else:
            server_settings = parse_model_from_args(ServerSettings, args)
            model_settings = [parse_model_from_args(ModelSettings, args)]
    except Exception as e:
        print(e, file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    assert server_settings is not None
    assert model_settings is not None
    app = create_app(
        server_settings=server_settings,
        model_settings=model_settings,
    )
    uvicorn.run(
        app,
        host=os.getenv("HOST", server_settings.host),
        port=int(os.getenv("PORT", server_settings.port)),
        ssl_keyfile=server_settings.ssl_keyfile,
        ssl_certfile=server_settings.ssl_certfile,
        limit_concurrency=10
    )


if __name__ == "__main__":
    main()
