import sys
import ctypes
import logging

import llama_cpp

# enum ggml_log_level {
#     GGML_LOG_LEVEL_NONE  = 0,
#     GGML_LOG_LEVEL_INFO  = 1,
#     GGML_LOG_LEVEL_WARN  = 2,
#     GGML_LOG_LEVEL_ERROR = 3,
#     GGML_LOG_LEVEL_DEBUG = 4,
#     GGML_LOG_LEVEL_CONT  = 5, // continue previous log
# };
GGML_LOG_LEVEL_TO_LOGGING_LEVEL = {
    0: logging.CRITICAL,
    1: logging.INFO,
    2: logging.WARNING,
    3: logging.ERROR,
    4: logging.DEBUG,
    5: logging.DEBUG,
}

logger = logging.getLogger("llama-cpp-python")

_last_log_level = GGML_LOG_LEVEL_TO_LOGGING_LEVEL[0]

# typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
@llama_cpp.llama_log_callback
def llama_log_callback(
    level: int,
    text: bytes,
    user_data: ctypes.c_void_p,
):
    # TODO: Correctly implement continue previous log
    global _last_log_level
    log_level = GGML_LOG_LEVEL_TO_LOGGING_LEVEL[level] if level != 5 else _last_log_level
    if logger.level <= GGML_LOG_LEVEL_TO_LOGGING_LEVEL[level]:
        print(text.decode("utf-8"), end="", flush=True, file=sys.stderr)
    _last_log_level = log_level


llama_cpp.llama_log_set(llama_log_callback, ctypes.c_void_p(0))


def set_verbose(verbose: bool):
    logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
