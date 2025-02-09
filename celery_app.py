from celery import Celery
import os
import time

# Tạo Celery App
celery_app = Celery(
    'tasks',
    broker=os.getenv('REDIS_URL', 'redis://redis:6379/0'),  # Broker URL cho Redis
    backend=os.getenv('REDIS_URL', 'redis://redis:6379/0')  # Backend URL cho lưu trữ kết quả
)

# Tác vụ nền nặng được xử lý bởi Celery Worker
@celery_app.task
def long_running_task():
    # Giả lập tác vụ nặng bằng việc "sleep"
    print("Đang chạy tác vụ nặng...")
    time.sleep(10)  # Giả lập tác vụ mất 10 giây để hoàn thành
    return "Hoàn thành tác vụ nặng!"
