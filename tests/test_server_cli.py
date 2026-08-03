import argparse

from pydantic import BaseModel, Field

from llama_cpp.server.cli import add_args_from_model


class ServerCliSettings(BaseModel):
    enabled: bool = Field(default=False, description="Enable feature")
    retries: int = Field(default=0, description="Retry count")
    workers: int = Field(default=2, description="Worker count")
    required_value: str = Field(description="Required value")


def test_add_args_from_model_includes_falsey_defaults_in_help():
    parser = argparse.ArgumentParser()

    add_args_from_model(parser, ServerCliSettings)

    help_by_dest = {action.dest: action.help for action in parser._actions}
    assert help_by_dest["enabled"] == "Enable feature (default: False)"
    assert help_by_dest["retries"] == "Retry count (default: 0)"
    assert help_by_dest["workers"] == "Worker count (default: 2)"
    assert help_by_dest["required_value"] == "Required value"
