FROM python:3.10-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    python3-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir faiss-cpu==1.7.4 --index-url https://pypi.org/simple
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data faiss_index

ENV FLASK_APP=app.py
ENV PORT=5000

EXPOSE $PORT

# Tăng timeout cho worker
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--timeout", "300", "--workers", "1", "app:app"]
