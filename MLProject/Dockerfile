FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY modelling.py .
COPY credit_risk_dataset_preprocessed/ credit_risk_dataset_preprocessed/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=200 mlflow scikit-learn pandas numpy

CMD ["python", "modelling.py"]
