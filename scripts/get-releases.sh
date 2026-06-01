#!/bin/bash

# Function to get all releases
get_all_releases() {
    local page=1
    local per_page=100
    local releases=""
    local new_releases

    # Prepare headers
    local headers=(-H "Accept: application/vnd.github.v3+json")
    if [ -n "$GITHUB_TOKEN" ]; then
        headers+=(-H "Authorization: Bearer $GITHUB_TOKEN")
    fi

    while true; do
        response=$(curl -s "${headers[@]}" \
                        "https://api.github.com/repos/abetlen/llama-cpp-python/releases?page=$page&per_page=$per_page")
        
        # Check if the response is valid JSON
        if ! echo "$response" | jq empty > /dev/null 2>&1; then
            echo "Error: Invalid response from GitHub API" >&2
            echo "Response: $response" >&2
            return 1
        fi

        new_releases=$(echo "$response" | jq -r '.[].tag_name')
        if [ -z "$new_releases" ]; then
            break
        fi
        releases="$releases $new_releases"
        ((page++))
    done

    echo $releases
}

# Get all releases and save to file
releases=$(get_all_releases)
if [ $? -ne 0 ]; then
    echo "Failed to fetch releases. Please check your internet connection and try again later." >&2
    exit 1
fi

echo "$releases" | tr ' ' '\n' > all_releases.txt

echo "All releases have been saved to all_releases.txt"
