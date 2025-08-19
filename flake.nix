{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Keep this identical to the version used by the llama.cpp git module
    llamaCppFlake.url = "github:ggerganov/llama.cpp/d924522a46c5ef097af4a88087d91673e8e87e4d";
  };
  outputs = { self, nixpkgs, flake-utils, llamaCppFlake }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        llamaCpp = llamaCppFlake.packages.${system}.default;
        pkgs = import nixpkgs { inherit system; };
        inherit (pkgs.stdenv) isDarwin;
        poetryEnv =  pkgs.poetry2nix.mkPoetryEnv {
          projectDir = self;
          python = pkgs.python310;
          extras = [ "server" ];
          groups = [ "dev" ];
          preferWheels = true;
          overrides = let
            extraDepsNeeded = {
              mkdocstrings-python = (self: super: null); # FIXME: recursion error in poetry2nix needs running down
            };
          in (pkgs.poetry2nix.defaultPoetryOverrides.extend (self: super:
            builtins.mapAttrs (package: build-requirements:
              if builtins.typeOf build-requirements == "list"
              then (builtins.getAttr package super).overridePythonAttrs (old: {
                buildInputs = (old.buildInputs or []) ++ (builtins.map (pkg: if builtins.isString pkg then builtins.getAttr pkg self else pkg) build-requirements);
              })
              else (build-requirements self super)
            ) extraDepsNeeded));
        };
      in {
        # packages.default = pkgs.poetry2nix.
        devShells.default = poetryEnv.env.overrideAttrs (oldAttrs: {
          LLAMA_CPP_LIB = "${llamaCpp}/lib/libllama.${if isDarwin then "dylib" else "so"}";
        });
        devShells.poetry = pkgs.mkShell {
          buildInputs = [
            llamaCpp
            pkgs.poetry
            pkgs.poetry2nix.cli
            pkgs.python310
          ];
        };
      });
}
