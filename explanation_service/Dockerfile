FROM python:3.12-slim

WORKDIR /app

COPY explanation_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY explanation_service/ ./explanation_service/
COPY shared/ ./shared/

CMD ["python", "-m", "explanation_service.consumer"]
