FROM python:3.10-slim

WORKDIR /app

# Cài đặt các dependencies cần thiết cho việc build
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Cài đặt Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data faiss_index

# Set environment variables
ENV FLASK_APP=app.py
ENV PORT=5000

# Expose the port
EXPOSE $PORT

# Run with gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT app:app
