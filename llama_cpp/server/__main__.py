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
import os
import sys
import argparse
from typing import List, Literal, Union

import uvicorn

from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings, ServerSettings, set_settings

EXE_NAME = 'llama_server'

def get_base_type(annotation):
    if getattr(annotation, '__origin__', None) is Literal:
        return type(annotation.__args__[0])
    elif getattr(annotation, '__origin__', None) is Union:
        non_optional_args = [arg for arg in annotation.__args__ if arg is not type(None)]
        if non_optional_args:
            return get_base_type(non_optional_args[0])
    elif getattr(annotation, '__origin__', None) is list or getattr(annotation, '__origin__', None) is List:
        return get_base_type(annotation.__args__[0])
    else:
        return annotation

def contains_list_type(annotation) -> bool:
    origin = getattr(annotation, '__origin__', None)
    
    if origin is list or origin is List:
        return True
    elif origin in (Literal, Union):
        return any(contains_list_type(arg) for arg in annotation.__args__)
    else:
        return False

def parse_bool_arg(arg):
    if isinstance(arg, bytes):
        arg = arg.decode('utf-8')

    true_values = {'1', 'on', 't', 'true', 'y', 'yes'}
    false_values = {'0', 'off', 'f', 'false', 'n', 'no'}

    arg_str = str(arg).lower().strip()
    
    if arg_str in true_values:
        return True
    elif arg_str in false_values:
        return False
    else:
        raise ValueError(f'Invalid boolean argument: {arg}')

def main():
    description = "🦙 Llama.cpp python server. Host your own LLMs!🚀"
    parser = argparse.ArgumentParser(EXE_NAME, description=description)
    for name, field in (ServerSettings.model_fields|Settings.model_fields).items():
        description = field.description
        if field.default and description and not field.is_required():
            description += f" (default: {field.default})"
        base_type = get_base_type(field.annotation) if field.annotation is not None else str
        list_type = contains_list_type(field.annotation)
        if base_type is not bool:
            parser.add_argument(
                f"--{name}",
                dest=name,
                nargs="*" if list_type else None,
                type=base_type,
                help=description,
            )
        if base_type is bool:
            parser.add_argument(
                f"--{name}",
                dest=name,
                type=parse_bool_arg,
                help=f"{description}",
            )
    
    try:
        args = parser.parse_args()
        server_settings = ServerSettings(**{k: v for k, v in vars(args).items() if v is not None})
        set_settings(server_settings)
        if server_settings.config and os.path.exists(server_settings.config):
            with open(server_settings.config, 'rb') as f:
                llama_settings = Settings.model_validate_json(f.read())
        else:
            llama_settings = Settings(**{k: v for k, v in vars(args).items() if v is not None})
        app = create_app(settings=llama_settings)
    except Exception as e:
        print(e, file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    uvicorn.run(
        app, host=server_settings.host, port=server_settings.port
    )

if __name__ == "__main__":
    main()
