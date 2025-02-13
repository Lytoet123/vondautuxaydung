FROM python:3.10-slim

WORKDIR /app

# Cài đặt các dependencies cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    python3-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt
COPY requirements.txt .

# Cài đặt faiss-cpu riêng trước
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir faiss-cpu==1.7.4 --index-url https://pypi.org/simple

# Sau đó cài đặt các package còn lại
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Tạo thư mục cần thiết
RUN mkdir -p data faiss_index

ENV FLASK_APP=app.py
ENV PORT=5000

EXPOSE $PORT

CMD gunicorn --bind 0.0.0.0:$PORT app:app
