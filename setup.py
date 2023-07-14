from skbuild import setup

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="llama_cpp_python",
    description="A Python wrapper for llama.cpp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.71",
    author="Andrei Betlen",
    author_email="abetlen@gmail.com",
    license="MIT",
    package_dir={"llama_cpp": "llama_cpp", "llama_cpp.server": "llama_cpp/server"},
    packages=["llama_cpp", "llama_cpp.server"],
    install_requires=["typing-extensions>=4.5.0", "numpy>=1.20.0", "diskcache>=5.6.1"],
    extras_require={
        "server": ["uvicorn>=0.22.0", "fastapi>=0.100.0", "pydantic-settings>=2.0.1", "sse-starlette>=1.6.1"],
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
