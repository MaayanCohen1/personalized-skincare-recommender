# Personalized Skincare Recommender

Microservices pipeline: **API** (FastAPI) → **vision** → **matching** → **explanation** (CrewAI + RAG), with **RabbitMQ** and **Redis**.

## Quick start

1. Copy env: `cp .env.example .env`
2. Run stack: `docker compose up --build`

## URLs

| URL | What |
|-----|------|
| http://localhost:8000/ | Demo UI |
| http://localhost:8000/docs | Swagger |
| http://localhost:15672 | RabbitMQ (guest / guest) |

## Try the API

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"skin_profile": {"skin_type": "oily", "skin_conditions": ["acne"]}}'

curl http://localhost:8000/result/<request_id>
```

## Layout & tests

- Folder tree: [STRUCTURE.md](STRUCTURE.md)
- Unit tests: `pytest explanation_service/tests matching_service/tests api_service/tests`
- Integration (Docker): `pytest integration_tests/ -m integration`

**Explanation KB (dev):** rebuild embeddings with `python -m explanation_service.init_rag_kb` inside the explanation service environment if you change markdown or ingredient JSON.
