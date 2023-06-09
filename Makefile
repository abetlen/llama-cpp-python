update:
	poetry install
	python3 -m pip install --upgrade pip
	git submodule update --init --recursive

update.vendor:
	cd vendor/llama.cpp && git pull origin master

build:
	python3 -m pip install --upgrade pip
	python3 -m pip install --verbose --editable .

build.cuda:
	python3 -m pip install --upgrade pip
	CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 python3 -m pip install --verbose --editable .

build.opencl:
	python3 -m pip install --upgrade pip
	CMAKE_ARGS="-DLLAMA_CLBLAST=on" FORCE_CMAKE=1 python3 -m pip install --verbose --editable . 

build.openblas:
	python3 -m pip install --upgrade pip
	CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1  python3 -m pip install --verbose --editable .

build.blis:
	python3 -m pip install --upgrade pip
	CMAKE_ARGS="-DLLAMA_OPENBLAS=on -DLLAMA_OPENBLAS_VENDOR=blis" FORCE_CMAKE=1  python3 -m pip install --verbose --editable .

build.metal:
	python3 -m pip install --upgrade pip
	CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1  python3 -m pip install --verbose --editable .

build.sdist:
	python3 -m pip install --upgrade pip build
	python3 -m build --sdist

deploy.pypi:
	python3 -m twine upload dist/*

deploy.gh-docs:
	mkdocs build
	mkdocs gh-deploy

clean:
	- cd vendor/llama.cpp && make clean
	- cd vendor/llama.cpp && rm libllama.so
	- rm llama_cpp/*.so
	- rm llama_cpp/*.dylib
	- rm llama_cpp/*.dll

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
	clean