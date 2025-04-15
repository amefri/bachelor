FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY check-bias-eo.py .
ENTRYPOINT ["python", "check-bias-eo.py"]