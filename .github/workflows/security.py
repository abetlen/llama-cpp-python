name: Bandit Security Scan

on:
  push:
    branches: [ "main" ]
  pull_request:
    # The branches below must be a subset of the branches above
    branches: [ "main" ]
  schedule:
    - cron: '30 3 * * *'

permissions:
  contents: read

jobs:
  bandit-security-scan:
    permissions:
      contents: read # for actions/checkout to fetch code
      security-events: write # for github/codeql-action/upload-sarif to upload SARIF results
      actions: read # only required for a private repository by github/codeql-action/upload-sarif to get the Action run status
    name: Bandit Security Scan
    runs-on: ubuntu-latest
    steps:
      # Checkout the repository to the GitHub Actions runner
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: 3.12

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install bandit[sarif]
          pip install .[all] -v

      # Execute Bandit
      - name: Run Bandit CLI
        run: |
          bandit -r . -v --format sarif -o results.sarif

      # Upload the SARIF file generated in the previous step
      - name: Upload SARIF results file
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
