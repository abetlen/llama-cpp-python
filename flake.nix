{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  # https://github.com/ggerganov/llama.cpp/pull/2277
  inputs.llama-cpp.url = "github:Freed-Wu/llama.cpp/flake";
  # inputs.llama-cpp.url = "github:abetlen/llama.cpp";
  outputs = { self, nixpkgs, flake-utils, llama-cpp }:
    flake-utils.lib.eachDefaultSystem
      (system:
        with nixpkgs.legacyPackages.${system};
        with python3.pkgs;
        {
          formatter = nixpkgs-fmt;
          packages.default = buildPythonApplication rec {
            name = "llama-cpp";
            src = self;
            format = "pyproject";
            disabled = pythonOlder "3.6";
            propagatedBuildInputs = [
              diskcache
              numpy
              typing-extensions
              fastapi
              # pydantic-settings
              # sse-starlette
              uvicorn
            ];
            nativeBuildInputs = [
              # https://github.com/abetlen/llama-cpp-python/pull/499
              scikit-build
              # scikit-build-core
            ];
            pythonImportsCheck = [
              "llama_cpp"
            ];
          };
        }
      );
}
