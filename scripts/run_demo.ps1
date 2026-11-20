$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
uv sync --frozen --extra dev
uv run finnews demo
uv run streamlit run app.py
