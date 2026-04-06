FROM python:3.12-slim

WORKDIR /app

COPY api_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api_service/ ./api_service/
COPY matching_service/core/ ./matching_service/core/
COPY shared/ ./shared/

CMD ["uvicorn", "api_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
