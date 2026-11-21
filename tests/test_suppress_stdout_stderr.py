import subprocess
import sys
from textwrap import dedent


def test_suppression_releases_null_streams_at_shutdown():
    script = dedent(
        """
        import gc
        import os
        import sys
        from llama_cpp._utils import suppress_stdout_stderr

        with suppress_stdout_stderr(disable=False):
            print("hidden", flush=True)
            with suppress_stdout_stderr(disable=False):
                os.write(2, b"hidden")
            os.write(1, b"hidden")

        print("restored", flush=True)
        print("restored", file=sys.stderr, flush=True)

        class Cleanup:
            def __del__(self):
                with suppress_stdout_stderr(disable=False):
                    os.write(2, b"hidden during shutdown")
                print("cleanup finished", file=sys.stderr, flush=True)

        gc.disable()
        cleanup = Cleanup()
        cleanup.cycle = cleanup
        """
    )
    result = subprocess.run(
        [sys.executable, "-Walways::ResourceWarning", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "restored\n"
    assert result.stderr == "restored\ncleanup finished\n"
