FROM python:3.9-slim

WORKDIR /app

# Copy requirements file first (better caching)
COPY requirements.txt .

# Install dependencies with no cache (reduce image size)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=200 --retries=10 \
    mlflow scikit-learn pandas numpy

# Copy source code after dependencies (avoid re-installing when code changes)
COPY modelling.py .
COPY credit_risk_dataset_preprocessed/ credit_risk_dataset_preprocessed/

# Create directory for mlruns
RUN mkdir -p /app/mlruns

CMD ["python", "modelling.py"]