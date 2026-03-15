# Alert Incident Intelligence

## Quick start
- Backend: `PYTHONPATH=$(pwd) uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000`
- Frontend: `cd frontend && npm install && npm run dev -- --host`
- Default API base used by the UI: http://localhost:8000

## Environment
- Copy `.env.example` to `.env` (or create `.env`) at repo root with Postgres creds. Minimum variables if you do not pass `db_url` in requests:
	- `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD` (or a single `DATABASE_URL`)
- If no DB vars are set, the API automatically falls back to a local SQLite file `local_alerts.db` in the repo root for demos.
- Data ships in `synthesis/` for the chatbot and `stitched_alerts_dedup.csv` for quick ingestion.

## API endpoints
- `POST /ingest` — normalize + dedupe vendor payloads into Postgres tables (`stitched_alerts_dedup`, etc.).
- `GET /alerts` — list deduped alerts (supports `limit`, `offset`, `table`, `schema`, `db_url`).
- `GET /alerts/ml` — stitched alerts with incident joins.
- `GET /alerts/severity` — severity counts; `GET /alerts/device` — device counts.
- `POST /chat` — chatbot grounded on synthetic vendor alerts.

## Deployment notes
- Python 3.13+, dependencies from `pyproject.toml` (`pip install -e .` works fine).
- Ensure the repo root stays on `PYTHONPATH` (the backend boot script inserts it to avoid clashing with stdlib `parser`).
- Run the frontend build with `npm run build` and serve `frontend/dist` behind your preferred static server.

