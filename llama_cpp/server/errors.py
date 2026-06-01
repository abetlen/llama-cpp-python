from __future__ import annotations

import sys
import traceback
import time
from re import compile, Match, Pattern
from typing import Callable, Coroutine, Optional, Tuple, Union, Dict
from typing_extensions import TypedDict


from fastapi import (
    Request,
    Response,
    HTTPException,
)
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from llama_cpp.server.types import (
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    CreateChatCompletionRequest,
)


class ErrorResponse(TypedDict):
    """OpenAI style error response"""

    message: str
    type: str
    param: Optional[str]
    code: Optional[str]


class ErrorResponseFormatters:
    """Collection of formatters for error responses.

    Args:
        request (Union[CreateCompletionRequest, CreateChatCompletionRequest]):
            Request body
        match (Match[str]): Match object from regex pattern

    Returns:
        Tuple[int, ErrorResponse]: Status code and error response
    """

    @staticmethod
    def context_length_exceeded(
        request: Union["CreateCompletionRequest", "CreateChatCompletionRequest"],
        match,  # type: Match[str] # type: ignore
    ) -> Tuple[int, ErrorResponse]:
        """Formatter for context length exceeded error"""

        context_window = int(match.group(2))
        prompt_tokens = int(match.group(1))
        completion_tokens = request.max_tokens
        if hasattr(request, "messages"):
            # Chat completion
            message = (
                "This model's maximum context length is {} tokens. "
                "However, you requested {} tokens "
                "({} in the messages, {} in the completion). "
                "Please reduce the length of the messages or completion."
            )
        else:
            # Text completion
            message = (
                "This model's maximum context length is {} tokens, "
                "however you requested {} tokens "
                "({} in your prompt; {} for the completion). "
                "Please reduce your prompt; or completion length."
            )
        return 400, ErrorResponse(
            message=message.format(
                context_window,
                (completion_tokens or 0) + prompt_tokens,
                prompt_tokens,
                completion_tokens,
            ),  # type: ignore
            type="invalid_request_error",
            param="messages",
            code="context_length_exceeded",
        )

    @staticmethod
    def model_not_found(
        request: Union["CreateCompletionRequest", "CreateChatCompletionRequest"],
        match,  # type: Match[str] # type: ignore
    ) -> Tuple[int, ErrorResponse]:
        """Formatter for model_not_found error"""

        model_path = str(match.group(1))
        message = f"The model `{model_path}` does not exist"
        return 400, ErrorResponse(
            message=message,
            type="invalid_request_error",
            param=None,
            code="model_not_found",
        )


class RouteErrorHandler(APIRoute):
    """Custom APIRoute that handles application errors and exceptions"""

    # key: regex pattern for original error message from llama_cpp
    # value: formatter function
    pattern_and_formatters: Dict[
        "Pattern[str]",
        Callable[
            [
                Union["CreateCompletionRequest", "CreateChatCompletionRequest"],
                "Match[str]",
            ],
            Tuple[int, ErrorResponse],
        ],
    ] = {
        compile(
            r"Requested tokens \((\d+)\) exceed context window of (\d+)"
        ): ErrorResponseFormatters.context_length_exceeded,
        compile(
            r"Model path does not exist: (.+)"
        ): ErrorResponseFormatters.model_not_found,
    }

    def error_message_wrapper(
        self,
        error: Exception,
        body: Optional[
            Union[
                "CreateChatCompletionRequest",
                "CreateCompletionRequest",
                "CreateEmbeddingRequest",
            ]
        ] = None,
    ) -> Tuple[int, ErrorResponse]:
        """Wraps error message in OpenAI style error response"""
        if body is not None and isinstance(
            body,
            (
                CreateCompletionRequest,
                CreateChatCompletionRequest,
            ),
        ):
            # When text completion or chat completion
            for pattern, callback in self.pattern_and_formatters.items():
                match = pattern.search(str(error))
                if match is not None:
                    return callback(body, match)

        # Only print the trace on unexpected exceptions
        print(f"Exception: {str(error)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        # Wrap other errors as internal server error
        return 500, ErrorResponse(
            message=str(error),
            type="internal_server_error",
            param=None,
            code=None,
        )

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[None, None, Response]]:
        """Defines custom route handler that catches exceptions and formats
        in OpenAI style error response"""

        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                start_sec = time.perf_counter()
                response = await original_route_handler(request)
                elapsed_time_ms = int((time.perf_counter() - start_sec) * 1000)
                response.headers["openai-processing-ms"] = f"{elapsed_time_ms}"
                return response
            except HTTPException as unauthorized:
                # api key check failed
                raise unauthorized
            except Exception as exc:
                json_body = await request.json()
                try:
                    if "messages" in json_body:
                        # Chat completion
                        body: Optional[
                            Union[
                                CreateChatCompletionRequest,
                                CreateCompletionRequest,
                                CreateEmbeddingRequest,
                            ]
                        ] = CreateChatCompletionRequest(**json_body)
                    elif "prompt" in json_body:
                        # Text completion
                        body = CreateCompletionRequest(**json_body)
                    else:
                        # Embedding
                        body = CreateEmbeddingRequest(**json_body)
                except Exception:
                    # Invalid request body
                    body = None

                # Get proper error message from the exception
                (
                    status_code,
                    error_message,
                ) = self.error_message_wrapper(error=exc, body=body)
                return JSONResponse(
                    {"error": error_message},
                    status_code=status_code,
                )

        return custom_route_handler
