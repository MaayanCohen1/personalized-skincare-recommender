FROM python:3.12-slim

WORKDIR /app

ENV CI=true \
    CREWAI_TELEMETRY_OPTOUT=true \
    OTEL_SDK_DISABLED=true

COPY explanation_service/requirements.txt ./requirements.txt
COPY explanation_service/requirements-ml.txt ./requirements-ml.txt
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

COPY explanation_service/ ./explanation_service/
COPY shared/ ./shared/

RUN python -m explanation_service.init_rag_kb

CMD ["python", "-m", "explanation_service.consumer"]