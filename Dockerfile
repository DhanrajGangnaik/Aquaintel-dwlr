# ========= Base image with Python + system libs =========
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System packages helpful for scientific/python plotting stacks
# - libgomp1: OpenMP runtime (scikit-learn, numpy)
# - fonts-dejavu-core: default fonts for matplotlib/plotly figures
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ========= Dependency layer =========
FROM base AS deps
# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ========= Test stage (used by CI) =========
# This stage fails the build if tests (or smoke import) fail.
FROM deps AS test
WORKDIR /app
COPY . .

# Ensure pytest is present even if not in requirements.txt
RUN pip install --no-cache-dir pytest

# Run real tests if found, otherwise do a smoke import of app & key libs
RUN python - <<'PY'
import sys, pathlib, importlib.util, subprocess

tests = list(pathlib.Path("tests").rglob("test_*.py"))
if tests:
    # Run pytest quietly; nonzero exit -> build fails
    raise SystemExit(subprocess.call(["pytest", "-q"]))
else:
    # Smoke checks: import critical modules and your app entry
    for mod in ["streamlit", "pandas", "numpy", "plotly", "sklearn", "requests"]:
        assert importlib.util.find_spec(mod) is not None, f"Missing {mod}"
    import importlib
    importlib.import_module("app")
    print("Smoke import OK")
PY

# ========= Runtime image =========
FROM deps AS runtime
WORKDIR /app
COPY . .

# Streamlit runs on 8501; app.py is the entry
EXPOSE 8501

# Optional: make Streamlit a bit quieter and bind to 0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Note: App references a sample CSV "DWLR_Dataset_2023.csv".
# Put it in the repo or mount at runtime: -v /path/to/data:/app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
