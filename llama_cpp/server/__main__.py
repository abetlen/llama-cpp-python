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
import argparse
from typing import List, Literal, Union

import uvicorn

from llama_cpp.server.app import create_app, Settings

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name, field in Settings.model_fields.items():
        description = field.description
        if field.default is not None and description is not None:
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
                action="store_true",
                help=f"Disable {description}",
                default=field.default,
            )
            parser.add_argument(
                f"--no-{name}",
                dest=name,
                action="store_false",
                help=f"Disable {description}",
                default=field.default,
            )

    args = parser.parse_args()
    settings = Settings(**{k: v for k, v in vars(args).items() if v is not None})
    app = create_app(settings=settings)

    uvicorn.run(
        app, host=os.getenv("HOST", settings.host), port=int(os.getenv("PORT", settings.port))
    )
