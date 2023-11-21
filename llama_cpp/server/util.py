import os
import shutil

def remove_file(path: str) -> None:
    if path and os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)

def models_root_dir(path = None):
    path = os.path.abspath(path or os.environ.get('MODEL', '/models'))
    if os.path.isdir(path): return path
    return os.path.dirname(path)
