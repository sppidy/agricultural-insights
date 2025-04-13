FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY train.py .
COPY dataset_dl.py .

# Create directories for datasets and models
RUN mkdir -p datasets models

# Copy datasets and models directories
COPY datasets/ ./datasets/
COPY models/ ./models/

EXPOSE 5010

CMD ["python", "app.py"]