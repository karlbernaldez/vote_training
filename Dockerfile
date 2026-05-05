FROM python:3.11-slim

WORKDIR /app

# Avoid Python writing .pyc files and keep logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install basic OS dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for better Docker layer caching
COPY Scripts/Homework/4/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy ingestion source code
COPY Scripts/Homework/4/ /app/

# Default command
CMD ["python", "ingest.py"]