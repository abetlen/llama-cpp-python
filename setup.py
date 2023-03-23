import os
import subprocess
from setuptools import setup, Extension

from distutils.command.build_ext import build_ext


class build_ext_custom(build_ext):
    def run(self):
        build_dir = os.path.join(os.getcwd(), "build")
        src_dir = os.path.join(os.getcwd(), "vendor", "llama.cpp")

        os.makedirs(build_dir, exist_ok=True)

        cmake_flags = [
            "-DLLAMA_STATIC=Off",
            "-DBUILD_SHARED_LIBS=On",
            "-DCMAKE_CXX_FLAGS=-fPIC",
            "-DCMAKE_C_FLAGS=-fPIC",
        ]
        subprocess.check_call(["cmake", src_dir, *cmake_flags], cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)

        # Move the shared library to the root directory
        lib_path = os.path.join(build_dir, "libllama.so")
        target_path = os.path.join(os.getcwd(), "libllama.so")
        os.rename(lib_path, target_path)


setup(
    name="llama_cpp",
    description="A Python wrapper for llama.cpp",
    version="0.0.1",
    author="Andrei Betlen",
    author_email="abetlen@gmail.com",
    license="MIT",
    py_modules=["llama_cpp"],
    ext_modules=[
        Extension("libllama", ["vendor/llama.cpp"]),
    ],
    cmdclass={"build_ext": build_ext_custom},
)
