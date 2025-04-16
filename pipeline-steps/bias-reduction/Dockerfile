FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bias-reduction.py .
ENTRYPOINT ["python", "bias-reduction.py"]