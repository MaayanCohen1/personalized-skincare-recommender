FROM python:3.12-slim

WORKDIR /app

COPY matching_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY matching_service/ ./matching_service/
COPY shared/ ./shared/

CMD ["python", "-m", "matching_service.worker"]
