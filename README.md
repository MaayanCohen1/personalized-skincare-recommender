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

The demo UI uses **multipart** `POST /submit` (image upload + questionnaire), polls `GET /result/{request_id}` every 2s, and expects the server to forward jobs to **`local_bridge`** (see below). Legacy JSON `POST /recommend` with an absolute `image_path` is still available for local tooling.

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"image_path":"/absolute/path/to/face.jpg","user_preferences":{"skin_type":"oily","has_breakouts":true,"sensitivities":["none"]}}'

curl http://localhost:8000/result/<request_id>
```

## Azure API + local workers (demo bridge)

Goal: keep **`api_service` on Azure App Service** while **RabbitMQ, Redis, vision, matching, explanation, and `local_bridge`** run on your machine (or your usual Docker stack). The browser talks only to Azure; Azure forwards the upload to your tunnel; when the pipeline finishes, **`local_bridge` POSTs** the result to Azure’s callback URL.

1. **Start local stack** (RabbitMQ, Redis, vision, matching, explanation) as you do today, e.g. `docker compose up --build`.
2. **Install and run `local_bridge`** from the repo root (needs the same `RABBITMQ_URL` your workers use, e.g. `amqp://guest:guest@localhost:5672/`):

   ```bash
   pip install -r local_bridge/requirements.txt
   export RABBITMQ_URL=amqp://guest:guest@localhost:5672/
   PYTHONPATH=. uvicorn local_bridge.main:app --host 0.0.0.0 --port 8090
   ```

3. **Expose port 8090** with a public tunnel (ngrok, Cloudflare Tunnel, etc.). Note the HTTPS origin (no trailing slash), e.g. `https://abc123.ngrok.app`.
4. **Azure App Service** application settings (names must match):

   | Setting | Example |
   |---------|---------|
   | `LOCAL_BRIDGE_URL` | `https://abc123.ngrok.app` |
   | `RESULT_CALLBACK_URL` | `https://<your-app>.azurewebsites.net/internal/result-callback` |
   | `CALLBACK_TOKEN` | Long random string; same value is forwarded to the bridge so it can authenticate callbacks |
   | `REDIS_URL` | Optional: still used for legacy `/recommend` + `GET /result` polling via Redis |
   | `ENABLE_RABBIT_RESULT_CONSUMER` | Set **`false`** on Azure if RabbitMQ is not reachable from App Service (recommended for bridge-only demos) |
   | `RABBITMQ_URL` | Required locally and for Docker; on Azure only if you keep the consumer enabled |

5. Open the Azure site root `/`, upload an image, submit, and wait for polling to show the routine.

**Flow:** Browser → Azure `POST /submit` → Azure enqueues state and forwards multipart to `{LOCAL_BRIDGE_URL}/demo-submit` → bridge writes a temp file, publishes `image.uploaded`, waits for `routine.completed` → bridge `POST`s JSON to `RESULT_CALLBACK_URL` with header `X-Callback-Token: <CALLBACK_TOKEN>` → browser sees `completed` on `GET /result/{request_id}`.

## Layout & tests

- Folder tree: [STRUCTURE.md](STRUCTURE.md)
- Unit tests: `PYTHONPATH=. pytest explanation_service/tests matching_service/tests api_service/tests local_bridge/tests`
- Integration (Docker): `pytest integration_tests/ -m integration`

**Explanation KB (dev):** rebuild embeddings with `python -m explanation_service.init_rag_kb` inside the explanation service environment if you change markdown or ingredient JSON.
