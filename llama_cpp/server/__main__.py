"""llama-cpp-python server from scratch in a single file.
"""

# import llama_cpp

# path = b"../../models/Qwen1.5-0.5B-Chat-GGUF/qwen1_5-0_5b-chat-q8_0.gguf"

# model_params = llama_cpp.llama_model_default_params()
# model = llama_cpp.llama_load_model_from_file(path, model_params)

# if model is None:
#     raise RuntimeError(f"Failed to load model from file: {path}")


# ctx_params = llama_cpp.llama_context_default_params()
# ctx = llama_cpp.llama_new_context_with_model(model, ctx_params)

# if ctx is None:
#     raise RuntimeError("Failed to create context")


from __future__ import annotations

import os
import sys
import argparse

import hypercorn.asyncio

from llama_cpp.server.app import create_app
from llama_cpp.server.settings import (
    Settings,
    ServerSettings,
    ModelSettings,
    ConfigFileSettings,
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
    server_settings: ServerSettings | None = None
    model_settings: list[ModelSettings] = []
    args = parser.parse_args()
    try:
        # Load server settings from config_file if provided
        config_file = os.environ.get("CONFIG_FILE", args.config_file)
        if config_file:
            if not os.path.exists(config_file):
                raise ValueError(f"Config file {config_file} not found!")
            with open(config_file, "rb") as f:
                # Check if yaml file
                if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                    import yaml
                    import json

                    config_file_settings = ConfigFileSettings.model_validate_json(
                        json.dumps(yaml.safe_load(f))
                    )
                else:
                    config_file_settings = ConfigFileSettings.model_validate_json(f.read())
                server_settings = ServerSettings.model_validate(config_file_settings)
                model_settings = config_file_settings.models
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
    config = hypercorn.Config()
    config.bind = [f"{os.getenv('HOST', server_settings.host)}:{int(os.getenv('PORT', server_settings.port))}"]
    config.ssl_keyfile = server_settings.ssl_keyfile
    config.ssl_certfile = server_settings.ssl_certfile
    hypercorn.asyncio.serve(app, config)


if __name__ == "__main__":
    main()
