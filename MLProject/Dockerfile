FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY modelling.py .
COPY credit_risk_dataset_preprocessed/ credit_risk_dataset_preprocessed/

RUN pip install --upgrade pip
RUN pip install --default-timeout=200 --retries=10 mlflow scikit-learn pandas numpy

CMD ["python", "modelling.py"]
