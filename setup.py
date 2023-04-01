from skbuild import setup

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="llama_cpp_python",
    description="A Python wrapper for llama.cpp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.11",
    author="Andrei Betlen",
    author_email="abetlen@gmail.com",
    license="MIT",
    packages=["llama_cpp"],
    install_requires=[
        "typing-extensions>=4.5.0",
    ],
    python_requires=">=3.7",
)
