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
# };
GGML_LOG_LEVEL_TO_LOGGING_LEVEL = {
    0: logging.CRITICAL,
    1: logging.INFO,
    2: logging.WARNING,
    3: logging.ERROR,
    4: logging.DEBUG,
}

logger = logging.getLogger("llama-cpp-python")


# typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
@llama_cpp.llama_log_callback
def llama_log_callback(
    level: int,
    text: bytes,
    user_data: ctypes.c_void_p,
):
    if logger.level <= GGML_LOG_LEVEL_TO_LOGGING_LEVEL[level]:
        print(text.decode("utf-8"), end="", flush=True, file=sys.stderr)


llama_cpp.llama_log_set(llama_log_callback, ctypes.c_void_p(0))


def set_verbose(verbose: bool):
    logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
