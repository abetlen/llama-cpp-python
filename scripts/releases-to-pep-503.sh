#!/bin/bash

# Get output directory or default to index/whl/cpu
output_dir=${1:-"index/whl/cpu"}

# Create output directory
mkdir -p $output_dir

# Change to output directory
pushd $output_dir

# Create an index html file
echo "<!DOCTYPE html>" > index.html
echo "<html>" >> index.html
echo "  <head></head>" >> index.html
echo "  <body>" >> index.html
echo "    <a href=\"llama-cpp-python/\">llama-cpp-python</a>" >> index.html
echo "    <br>" >> index.html
echo "  </body>" >> index.html
echo "</html>" >> index.html
echo "" >> index.html

# Create llama-cpp-python directory
mkdir -p llama-cpp-python

# Change to llama-cpp-python directory
pushd llama-cpp-python

# Create an index html file
echo "<!DOCTYPE html>" > index.html
echo "<html>" >> index.html
echo "  <body>" >> index.html
echo "    <h1>Links for llama-cpp-python</h1>" >> index.html

# Get all releases
releases=$(curl -s https://api.github.com/repos/abetlen/llama-cpp-python/releases | jq -r .[].tag_name)

# Get pattern from second arg or default to valid python package version pattern
pattern=${2:-"^[v]?[0-9]+\.[0-9]+\.[0-9]+$"}

# Filter releases by pattern
releases=$(echo $releases | tr ' ' '\n' | grep -E $pattern)

# For each release, get all assets
for release in $releases; do
    assets=$(curl -s https://api.github.com/repos/abetlen/llama-cpp-python/releases/tags/$release | jq -r .assets)
    # Get release version from release ie v0.1.0-cu121 -> v0.1.0
    release_version=$(echo $release | grep -oE "^[v]?[0-9]+\.[0-9]+\.[0-9]+")
    echo "    <h2>$release_version</h2>" >> index.html
    for asset in $(echo $assets | jq -r .[].browser_download_url); do
        if [[ $asset == *".whl" ]]; then
            echo "    <a href=\"$asset\">$asset</a>" >> index.html
            echo "    <br>" >> index.html
        fi
    done
done

echo "  </body>" >> index.html
echo "</html>" >> index.html
echo "" >> index.html
