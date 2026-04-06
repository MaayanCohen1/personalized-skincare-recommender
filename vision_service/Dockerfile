FROM python:3.12-slim

WORKDIR /app

COPY vision_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY vision_service/ ./vision_service/
COPY shared/ ./shared/

CMD ["python", "-m", "vision_service.worker"]
