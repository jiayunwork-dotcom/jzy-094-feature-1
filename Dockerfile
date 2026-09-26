FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，利用层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷代码与测试
COPY app ./app
COPY tests ./tests
COPY pytest.ini ./pytest.ini

EXPOSE 8000

# 起容器即在固定端口 8000 听请求：
#   docker run --rm -p 8000:8000 <image>
# 在容器里执行测试：
#   docker run --rm <image> pytest -q
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
