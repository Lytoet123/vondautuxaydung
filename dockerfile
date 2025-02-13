FROM python:3.10-slim

# Cài đặt các dependencies hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Tạo và kích hoạt môi trường ảo
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Tạo thư mục làm việc
WORKDIR /app

# Copy và cài đặt requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Thiết lập biến môi trường cho Railway
ENV PORT=5000

# Expose port
EXPOSE $PORT

# Start command
CMD gunicorn --bind 0.0.0.0:$PORT app:app
