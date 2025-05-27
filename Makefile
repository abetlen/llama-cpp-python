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
	python3 -m pip install \
		--verbose \
		--config-settings=cmake.verbose=true \
		--config-settings=logging.level=INFO \
		--config-settings=install.strip=false  \
		--config-settings=cmake.args="-DCMAKE_BUILD_TYPE=Debug;-DCMAKE_C_FLAGS='-ggdb -O0';-DCMAKE_CXX_FLAGS='-ggdb -O0'" \
		--editable .

build.debug.extra:
	python3 -m pip install \
		--verbose \
		--config-settings=cmake.verbose=true \
		--config-settings=logging.level=INFO \
		--config-settings=install.strip=false  \
		--config-settings=cmake.args="-DCMAKE_BUILD_TYPE=Debug;-DCMAKE_C_FLAGS='-fsanitize=address -ggdb -O0';-DCMAKE_CXX_FLAGS='-fsanitize=address -ggdb -O0'" \
		--editable .

build.cuda:
	CMAKE_ARGS="-DGGML_CUDA=on" python3 -m pip install --verbose -e .

build.openblas:
	CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" python3 -m pip install --verbose -e .

build.blis:
	CMAKE_ARGS="-DGGML_BLAS=on -DGGML_BLAS_VENDOR=FLAME" python3 -m pip install --verbose -e .

build.metal:
	CMAKE_ARGS="-DGGML_METAL=on" python3 -m pip install --verbose -e .

build.vulkan:
	CMAKE_ARGS="-DGGML_VULKAN=on" python3 -m pip install --verbose -e .

build.kompute:
	CMAKE_ARGS="-DGGML_KOMPUTE=on" python3 -m pip install --verbose -e .

build.sycl:
	CMAKE_ARGS="-DGGML_SYCL=on" python3 -m pip install --verbose -e .

build.rpc:
	CMAKE_ARGS="-DGGML_RPC=on" python3 -m pip install --verbose -e .

build.sdist:
	python3 -m build --sdist --verbose

deploy.pypi:
	python3 -m twine upload dist/*

deploy.gh-docs:
	mkdocs build
	mkdocs gh-deploy

test:
	python3 -m pytest --full-trace -v

docker:
	docker build -t llama-cpp-python:latest -f docker/simple/Dockerfile .

run-server:
	python3 -m llama_cpp.server --model ${MODEL}

clean:
	- cd vendor/llama.cpp && make clean
	- cd vendor/llama.cpp && rm libllama.so
	- rm -rf _skbuild
	- rm llama_cpp/lib/*.so
	- rm llama_cpp/lib/*.dylib
	- rm llama_cpp/lib/*.metal
	- rm llama_cpp/lib/*.dll
	- rm llama_cpp/lib/*.lib

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
