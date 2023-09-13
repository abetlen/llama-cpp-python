# How to use this file
#
# 1. create a folder called "hooks" in your repo
# 2. copy this file there
# 3. add the --additional-hooks-dir flag to your pyinstaller command:
#    ex: `pyinstaller --name binary-name --additional-hooks-dir=./hooks entry-point.py`


from PyInstaller.utils.hooks import collect_data_files, get_package_paths
import os

# Get the package path
package_path = get_package_paths('llama_cpp')[0]

# Collect data files
datas = collect_data_files('llama_cpp')

# Append the additional .dll or .so file
if os.name == 'nt':  # Windows
    dll_path = os.path.join(package_path, 'llama_cpp', 'llama.dll')
    datas.append((dll_path, 'llama_cpp'))
elif os.name == 'posix':  # Linux/Mac
    so_path = os.path.join(package_path, 'llama_cpp', 'llama.so')
    datas.append((so_path, 'llama_cpp'))
