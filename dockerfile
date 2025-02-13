# Sử dụng image Python với đầy đủ build tools
FROM python:3.10-buster

# Cài đặt các dependencies hệ thống
RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    libopenblas-dev \
    python3-dev \
    swig \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy requirements.txt trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt các dependencies Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Expose port
EXPOSE 5000

# Chạy ứng dụng với Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
