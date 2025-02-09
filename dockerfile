# Sử dụng image Python với distutils
FROM python:3.10-slim-buster

# Cài đặt các dependencies hệ thống cần thiết (nếu có)
# Ví dụ: libmagic cho python-magic
# RUN apt-get update && apt-get install -y libmagic-dev --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy requirements.txt trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt các dependencies từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code của ứng dụng
COPY . .

# Thiết lập biến môi trường (nếu cần)
# ENV OPENAI_API_KEY=your_api_key

# Expose port (nếu cần)
EXPOSE 5000

# Lệnh chạy ứng dụng
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
