"""Example FastAPI server for llama.cpp.

To run this example:

```bash
pip install fastapi uvicorn sse-starlette pydantic-settings
export MODEL=../models/7B/...
```

Then run:
```
uvicorn llama_cpp.server.app:app --reload
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
    Server,
    ServerSettings,
    ModelSettings,
    ConfigFileSettings,
    set_server_settings,
)
from llama_cpp.server.cli import add_args_from_model, parse_model_from_args


def main():
    description = "🦙 Llama.cpp python server. Host your own LLMs!🚀"
    parser = argparse.ArgumentParser(description=description)

    add_args_from_model(parser, ModelSettings)
    add_args_from_model(parser, ServerSettings)
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to a config file to load.",
    )
    try:
        args = parser.parse_args()
        server_settings: ServerSettings | None = None
        model_settings: list[ModelSettings] = []
        # Load server settings from config_file if provided
        config_file = os.environ.get("CONFIG_FILE", args.config_file)
        if config_file:
            if not os.path.exists(config_file):
                raise ValueError(f"Config file {config_file} not found!")
            with open(config_file, "rb") as f:
                config_file_settings = ConfigFileSettings.model_validate_json(f.read())
                server_settings = ServerSettings(
                    **{
                        k: v
                        for k, v in config_file_settings.model_dump().items()
                        if k in ServerSettings.model_fields
                    }
                )
                model_settings = config_file_settings.models
        else:
            server_settings = ServerSettings(
                **{
                    k: v
                    for k, v in vars(args).items()
                    if k in ServerSettings.model_fields
                }
            )
            model_settings = [
                ModelSettings(
                    **{
                        k: v
                        for k, v in vars(args).items()
                        if k in ModelSettings.model_fields
                    }
                )
            ]
        app = create_app(
            settings=Server(
                **server_settings.model_dump(), **model_settings[0].model_dump()
            )
        )
        uvicorn.run(
            app,
            host=os.getenv("HOST", server_settings.host),
            port=int(os.getenv("PORT", server_settings.port)),
            ssl_keyfile=server_settings.ssl_keyfile,
            ssl_certfile=server_settings.ssl_certfile,
        )
    except Exception as e:
        print(e, file=sys.stderr)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
