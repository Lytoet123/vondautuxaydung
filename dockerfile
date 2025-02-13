FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data faiss_index

# Set environment variable for Flask
ENV FLASK_APP=app.py
ENV PORT=5000

# Expose the port
EXPOSE $PORT

# Run the application with gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT app:app
