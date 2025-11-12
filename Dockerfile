# Multi-stage build with testing gate
FROM python:3.12-slim AS deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM deps AS test
COPY . .
RUN pytest -q     # CI will fail here if tests fail

FROM deps AS runtime
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
