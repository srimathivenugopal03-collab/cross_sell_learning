# PowerShell Task Automation Script for Object Counter Service

param (
    [Parameter(Position = 0)]
    [ValidateSet("help", "dev", "run", "check", "typecheck", "install", "test", "test-cov", "lint", "format", "ruff", "run-dev", "run-prod", "docker-up", "docker-down", "setup-model", "clean")]
    [string]$Task = "help"
)

function Show-Help {
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host "  Object Counter Service - Task Automation (PowerShell)" -ForegroundColor Cyan
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host "Usage: .\run.ps1 <command>`n"
    Write-Host "Quick Run Commands:"
    Write-Host "  dev / run    - Start local API server in dev mode"
    Write-Host "  check        - Run all quality checks (lint + typecheck + tests)"
    Write-Host "  typecheck    - Run mypy static type analysis"
    Write-Host "`nCode Quality & Formatting Commands:"
    Write-Host "  lint         - Check code style and linting with Ruff"
    Write-Host "  format       - Auto-format code and sort imports with Ruff"
    Write-Host "  ruff         - Run Ruff formatting and lint auto-fix"
    Write-Host "  clean        - Clean all caches (.pytest_cache, .ruff_cache, .mypy_cache, __pycache__, .coverage)"
    Write-Host "`nAll Other Commands:"
    Write-Host "  install      - Install all dependencies using uv"
    Write-Host "  test         - Run unit & integration test suite (53 tests)"
    Write-Host "  test-cov     - Run tests with test coverage reporting"
    Write-Host "  run-prod     - Run API locally pointing to prod services"
    Write-Host "  docker-up    - Build and start full stack in Docker Compose"
    Write-Host "  docker-down  - Stop all Docker Compose containers"
    Write-Host "  setup-model  - Download and extract SSD MobileNet v2 model"
}

switch ($Task) {
    "dev" {
        Write-Host "--> Starting Object Counter in DEV mode (in-memory)..." -ForegroundColor Green
        $env:ENV = "dev"
        uv run python -m counter.entrypoints.webapp
    }
    "run" {
        Write-Host "--> Starting Object Counter in DEV mode (in-memory)..." -ForegroundColor Green
        $env:ENV = "dev"
        uv run python -m counter.entrypoints.webapp
    }
    "check" {
        Write-Host "--> 1/3: Checking linting & formatting with Ruff..." -ForegroundColor Green
        uv run ruff check counter tests
        uv run ruff format --check counter tests
        Write-Host "--> 2/3: Checking static types with Mypy..." -ForegroundColor Green
        uv run mypy counter
        Write-Host "--> 3/3: Running full test suite with Pytest..." -ForegroundColor Green
        uv run pytest
    }
    "typecheck" {
        Write-Host "--> Running mypy static type checking..." -ForegroundColor Green
        uv run mypy counter
    }
    "install" {
        Write-Host "--> Installing dependencies using uv..." -ForegroundColor Green
        uv sync --all-extras
    }
    "test" {
        Write-Host "--> Running test suite..." -ForegroundColor Green
        uv run pytest
    }
    "test-cov" {
        Write-Host "--> Running tests with coverage..." -ForegroundColor Green
        uv run pytest --cov=counter tests/
    }
    "lint" {
        Write-Host "--> Checking code style and linting with Ruff..." -ForegroundColor Green
        uv run ruff check counter tests
    }
    "format" {
        Write-Host "--> 1/2: Formatting code with Ruff..." -ForegroundColor Green
        uv run ruff format counter tests
        Write-Host "--> 2/2: Auto-fixing lint issues & import order with Ruff..." -ForegroundColor Green
        uv run ruff check --fix counter tests
        Write-Host "--> Code formatting complete!" -ForegroundColor Green
    }
    "ruff" {
        Write-Host "--> Running Ruff format and lint checks..." -ForegroundColor Green
        uv run ruff format counter tests
        uv run ruff check --fix counter tests
    }
    "run-dev" {
        Write-Host "--> Starting Object Counter in DEV mode (in-memory)..." -ForegroundColor Green
        $env:ENV = "dev"
        uv run python -m counter.entrypoints.webapp
    }
    "run-prod" {
        Write-Host "--> Starting Object Counter in PROD mode (PostgreSQL + TFS)..." -ForegroundColor Green
        $env:ENV = "prod"
        $env:DB_TYPE = "postgres"
        $env:DETECTOR_TYPE = "tfs"
        uv run python -m counter.entrypoints.webapp
    }
    "docker-up" {
        Write-Host "--> Launching Docker Compose stack..." -ForegroundColor Green
        docker compose up --build -d
    }
    "docker-down" {
        Write-Host "--> Stopping Docker Compose stack..." -ForegroundColor Green
        docker compose down -v
    }
    "setup-model" {
        Write-Host "--> Downloading SSD MobileNet v2 model..." -ForegroundColor Green
        New-Item -ItemType Directory -Force -Path "tmp/model/ssd_mobilenet_v2/1" | Out-Null
        $modelUrl = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
        $tarPath = "tmp/model.tar.gz"
        Invoke-WebRequest -Uri $modelUrl -OutFile $tarPath
        tar -xzvf $tarPath -C tmp/model
        Move-Item -Force "tmp/model/ssd_mobilenet_v2_coco_2018_03_29/saved_model/saved_model.pb" "tmp/model/ssd_mobilenet_v2/1/"
        Remove-Item -Force $tarPath
        Remove-Item -Recurse -Force "tmp/model/ssd_mobilenet_v2_coco_2018_03_29"
        Write-Host "--> Model setup complete in tmp/model/ssd_mobilenet_v2/1/saved_model.pb" -ForegroundColor Green
    }
    "clean" {
        Write-Host "--> Cleaning all cache directories and temporary files..." -ForegroundColor Green
        $cacheItems = @(".pytest_cache", ".ruff_cache", ".mypy_cache", "htmlcov", ".coverage", "coverage.xml", "*.egg-info", "build", "dist")
        foreach ($item in $cacheItems) {
            if (Test-Path $item) {
                Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $item
            }
        }
        if (Test-Path "tmp/debug") {
            Get-ChildItem -Path "tmp/debug" -Filter "*.jpg" -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path "counter") {
            Get-ChildItem -Path "counter" -Recurse -Filter "__pycache__" -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
            Get-ChildItem -Path "counter" -Recurse -Include "*.pyc", "*.pyo" -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path "tests") {
            Get-ChildItem -Path "tests" -Recurse -Filter "__pycache__" -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
            Get-ChildItem -Path "tests" -Recurse -Include "*.pyc", "*.pyo" -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
        }
        Write-Host "--> Cache files successfully cleaned!" -ForegroundColor Green
    }
    Default {
        Show-Help
    }
}

