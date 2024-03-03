#!/bin/bash

# Create an index html file
echo "<!DOCTYPE html>"
echo "<html>"
echo "  <body>"
echo "    <h1>Links for llama-cpp-python</h1>"

# Get all releases
releases=$(curl -s https://api.github.com/repos/abetlen/llama-cpp-python/releases | jq -r .[].tag_name)

# For each release, get all assets
for release in $releases; do
    assets=$(curl -s https://api.github.com/repos/abetlen/llama-cpp-python/releases/tags/$release | jq -r .assets)
    echo "    <h2>$release</h2>"
    for asset in $(echo $assets | jq -r .[].browser_download_url); do
        if [[ $asset == *".whl" ]]; then
            echo "    <a href=\"$asset\">$asset</a>"
            echo "    <br>"
        fi
    done
done

echo "  </body>"
echo "</html>"