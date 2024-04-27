update:
	poetry install
	git submodule update --init --recursive

update.vendor:
	cd vendor/llama.cpp && git pull origin master

deps:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[all]"

build:
	python3 -m pip install --verbose -e .

build.debug:
	CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug" python3 -m pip install --verbose --config-settings=cmake.verbose=true --config-settings=logging.level=INFO --config-settings=install.strip=false  --editable .

build.cuda:
	CMAKE_ARGS="-DLLAMA_CUBLAS=on" python3 -m pip install --verbose -e .

build.opencl:
	CMAKE_ARGS="-DLLAMA_CLBLAST=on" python3 -m pip install --verbose -e .

build.openblas:
	CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" python3 -m pip install --verbose -e .

build.blis:
	CMAKE_ARGS="-DLLAMA_BLAS=on -DLLAMA_BLAS_VENDOR=FLAME" python3 -m pip install --verbose -e .

build.metal:
	CMAKE_ARGS="-DLLAMA_METAL=on" python3 -m pip install --verbose -e .

build.vulkan:
	CMAKE_ARGS="-DLLAMA_VULKAN=on" python3 -m pip install --verbose -e .

build.kompute:
	CMAKE_ARGS="-DLLAMA_KOMPUTE=on" python3 -m pip install --verbose -e .

build.sycl:
	CMAKE_ARGS="-DLLAMA_SYCL=on" python3 -m pip install --verbose -e .

build.sdist:
	python3 -m build --sdist

deploy.pypi:
	python3 -m twine upload dist/*

deploy.gh-docs:
	mkdocs build
	mkdocs gh-deploy

test:
	python3 -m pytest

docker:
	docker build -t llama-cpp-python:latest -f docker/simple/Dockerfile .

run-server:
	uvicorn --factory llama.server:app --host ${HOST} --port ${PORT}

clean:
	- cd vendor/llama.cpp && make clean
	- cd vendor/llama.cpp && rm libllama.so
	- rm -rf _skbuild
	- rm llama_cpp/*.so
	- rm llama_cpp/*.dylib
	- rm llama_cpp/*.metal
	- rm llama_cpp/*.dll
	- rm llama_cpp/*.lib

.PHONY: \
	update \
	update.vendor \
	build \
	build.cuda \
	build.opencl \
	build.openblas \
	build.sdist \
	deploy.pypi \
	deploy.gh-docs \
	docker \
	clean