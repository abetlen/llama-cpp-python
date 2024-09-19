#!/bin/bash

# Enable exit on error
set -e

# Function for logging
log_error() {
    echo "ERROR: $1" >&2
}

log_info() {
    echo "INFO: $1"
}

# Get output directory or default to index/whl/cpu
output_dir=${1:-"index/whl/cpu"}

# Get pattern from second arg or default to valid python package version pattern
pattern=${2:-"^[v]?[0-9]+\.[0-9]+\.[0-9]+$"}

# Get the current directory (where the script is run from)
current_dir="$(pwd)"

# Check if all_releases.txt exists
if [ ! -f "$current_dir/all_releases.txt" ]; then
    log_error "all_releases.txt not found in the current directory."
    exit 1
fi

# Create output directory
mkdir -p "$output_dir"

# Create an index html file
cat << EOF > "$output_dir/index.html"
<!DOCTYPE html>
<html>
  <head></head>
  <body>
    <a href="llama-cpp-python/">llama-cpp-python</a>
    <br>
  </body>
</html>

EOF

# Create llama-cpp-python directory
mkdir -p "$output_dir/llama-cpp-python"

# Create an index html file in llama-cpp-python directory
cat << EOF > "$output_dir/llama-cpp-python/index.html"
<!DOCTYPE html>
<html>
  <body>
    <h1>Links for llama-cpp-python</h1>
EOF

# Filter releases by pattern
releases=$(grep -E "$pattern" "$current_dir/all_releases.txt")

# Prepare curl headers
headers=('--header' 'Accept: application/vnd.github.v3+json')
if [ -n "$GITHUB_TOKEN" ]; then
    headers+=('--header' "authorization: Bearer $GITHUB_TOKEN")
fi
headers+=('--header' 'content-type: application/json')

# For each release, get all assets
for release in $releases; do
    log_info "Processing release: $release"
    response=$(curl -s "${headers[@]}" \
                    "https://api.github.com/repos/abetlen/llama-cpp-python/releases/tags/$release")
    
    if [ -z "$response" ]; then
        log_error "Empty response from GitHub API for release $release"
        continue
    fi

    if ! echo "$response" | jq -e '.assets' > /dev/null 2>&1; then
        log_error "Invalid or unexpected response from GitHub API for release $release"
        log_error "Response: $response"
        continue
    fi

    # Get release version from release ie v0.1.0-cu121 -> v0.1.0
    release_version=$(echo "$release" | grep -oE "^[v]?[0-9]+\.[0-9]+\.[0-9]+")
    echo "    <h2>$release_version</h2>" >> "$output_dir/llama-cpp-python/index.html"
    
    wheel_urls=$(echo "$response" | jq -r '.assets[] | select(.name | endswith(".whl")) | .browser_download_url')
    if [ -z "$wheel_urls" ]; then
        log_error "No wheel files found for release $release"
        continue
    fi

    echo "$wheel_urls" | while read -r asset; do
        echo "    <a href=\"$asset\">$asset</a>" >> "$output_dir/llama-cpp-python/index.html"
        echo "    <br>" >> "$output_dir/llama-cpp-python/index.html"
    done
done

echo "  </body>" >> "$output_dir/llama-cpp-python/index.html"
echo "</html>" >> "$output_dir/llama-cpp-python/index.html"
echo "" >> "$output_dir/llama-cpp-python/index.html"

log_info "Index generation complete. Output directory: $output_dir"
