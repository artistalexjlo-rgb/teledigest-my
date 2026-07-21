FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# BRAIN_DIR (/brain) монтируется Dokploy: Directory Mount /root/pseo_builder -> /brain
CMD ["python", "-u", "bot.py"]
