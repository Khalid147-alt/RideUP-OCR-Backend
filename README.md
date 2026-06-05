---
title: RideUp OCR Backend
emoji: 🚗
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: UK delivery screenshot OCR via Gemini 2.5 Flash Vision
---

# RideUp OCR Backend

Production OCR API that extracts **pay, miles, minutes, orders, platform, and confidence** from UK delivery driver screenshots (Uber Eats + Deliveroo). Built on **FastAPI + Gemini 2.5 Flash Vision + Pydantic v2**, deployed on **HuggingFace Spaces**.

---

## Live Demo

- **API base URL** — https://khalid147-rideup-ocr-backend.hf.space
- **Swagger UI** — https://khalid147-rideup-ocr-backend.hf.space/docs
- **Health check** — https://khalid147-rideup-ocr-backend.hf.space/health

---

## Features

- 📸 **Multipart + base64 upload** — works from web file pickers, mobile cameras, and base64-encoded payloads alike.
- 🤖 **Gemini 2.5 Flash Vision** — fast, accurate, free-tier friendly OCR purpose-tuned for UK delivery UIs.
- 🎯 **Auto platform detection** — server-side teal-pixel signature routes Deliveroo V2 screenshots to a specialised prompt before they reach Gemini; everything else falls through to the master prompt.
- 📦 **Two Deliveroo layouts** — V1 (classic, with miles/minutes) and V2 (modern teal "Accept and go" cards where miles/minutes are intentionally absent).
- 🛡️ **Server-side confidence override** — confidence is recomputed from the fields actually present, not the model's self-claim.
- 📍 **Postcode mileage estimation (Deliveroo V2)** — when the offer card hides distance but prints pickup/drop-off postcodes, the backend geocodes them via the free [postcodes.io](https://postcodes.io) API and estimates the trip distance with the Haversine formula. Fully fail-silent: if the postcodes aren't there or the API is unreachable, `miles` simply stays `null`.
- 🔁 **Resilient by design** — 15s/30s/60s rate-limit backoff, single-retry JSON repair, and graceful 502 → HTTP 200 low-confidence degradation so the client never has to handle upstream blips.
- ✅ **109 tests passing** — extraction, rate-limit handling, resilience, postcode mileage estimation, and end-to-end fixtures.

---

## API Endpoints

| Endpoint          | Method | Description                                                                |
| ----------------- | ------ | -------------------------------------------------------------------------- |
| `/extract`        | POST   | Multipart image upload (JPG/PNG/WEBP). Returns structured JSON.            |
| `/extract/base64` | POST   | JSON body `{ "image": "<base64 or data: URL>" }`. Same response shape.     |
| `/health`         | GET    | Liveness probe: `{"status":"ok","model":"gemini-2.5-flash","version":"1.0.0"}`. |
| `/docs`           | GET    | Interactive Swagger UI (auto-generated).                                   |
| `/`               | GET    | API metadata and endpoint index.                                           |

### Example call

```bash
curl -X POST https://khalid147-rideup-ocr-backend.hf.space/extract \
  -F "image=@./screenshot.png"
```

---

## Output Schema

```json
{
  "pay": 4.23,
  "currency": "GBP",
  "miles": 3.2,
  "minutes": null,
  "orders": 1,
  "platform": "deliveroo",
  "confidence": "high",
  "pickup_postcode": "NW10 0NX",
  "dropoff_postcode": "HA9 6FF",
  "notes": "Deliveroo V2 layout. Miles estimated from postcodes NW10 0NX → HA9 6FF via postcodes.io. retry_attempted=false",
  "raw_text": "£4.23 · 1 order · McDonald's · 139 NORTH CIRCULAR ROAD NW100NX · 1 Harrow Road HA96FF · Accept and go"
}
```

When no postcodes are present (or estimation isn't applicable), `miles` is `null` and the two postcode fields are `null`:

```json
{
  "pay": 11.08,
  "currency": "GBP",
  "miles": null,
  "minutes": null,
  "orders": 3,
  "platform": "deliveroo",
  "confidence": "high",
  "pickup_postcode": null,
  "dropoff_postcode": null,
  "notes": "Deliveroo V2 layout. retry_attempted=false",
  "raw_text": "£11.08 · 3 orders · Accept and go"
}
```

| Field              | Type            | Description                                                                |
| ------------------ | --------------- | -------------------------------------------------------------------------- |
| `pay`              | `float \| null` | Trip earnings as a decimal. `null` when not visible.                        |
| `currency`         | `string`        | `"GBP"`, `"USD"`, `"EUR"`, or `"unknown"`.                                  |
| `miles`            | `float \| null` | Distance in miles. Native when shown; **estimated from postcodes for Deliveroo V2** (clearly labelled in `notes`); `null` when neither is available. |
| `minutes`          | `int \| null`   | Trip duration in minutes. **`null` is correct for Deliveroo V2**.           |
| `orders`           | `int \| null`   | Number of orders / stops in the batch.                                      |
| `platform`         | `string`        | `"uber_eats"`, `"deliveroo"`, or `"unknown"`.                               |
| `confidence`       | `string`        | `"high"` / `"medium"` / `"low"` — recomputed server-side from actual fields. Not downgraded by mileage estimation. |
| `pickup_postcode`  | `str \| null`   | Pickup postcode parsed from `raw_text` (Deliveroo V2). `null` when none found. |
| `dropoff_postcode` | `str \| null`   | Drop-off postcode parsed from `raw_text` (Deliveroo V2). `null` when none found. |
| `notes`            | `string`        | Layout detected, conversions applied, mileage-estimation marker, retry markers, edge cases. |
| `raw_text`         | `string`        | Raw OCR text the model read — useful for debugging and audit.               |

---

## Supported Platforms

### Uber Eats
Single-order and stacked-batch trip cards. Always returns `pay`, `miles`, `minutes`, and `orders` when visible.

### Deliveroo V1 (classic layout)
The older Deliveroo card shows miles and minutes inline. Behaves like Uber Eats — all four core fields populate.

### Deliveroo V2 (modern "Accept and go" layout)
The current Deliveroo card with the teal accent strip **does not display miles or minutes** — only `pay` and `orders`. This is **intentional**: the API returns `miles: null` and `minutes: null` for these screenshots, and confidence stays `high` because all available core fields were captured. Frontends should render "N/A" or "—" for these fields rather than treating `null` as a parse failure.

Platform detection is automatic — teal-pixel ratio ≥ 5% routes to the Deliveroo V2 prompt; everything else uses the master prompt.

---

## Postcode-Based Mileage Estimation (Deliveroo V2)

Deliveroo V2 cards never show distance to the driver, so a correct read returns `miles: null`. However, the card **does** print the pickup and drop-off addresses — including UK postcodes — inside `raw_text`. The backend recovers an estimated distance from those postcodes.

### How it works

1. **Extract** — both postcodes are parsed from `raw_text` with a UK-postcode regex and normalised to canonical form (a missing space is inserted: `NW100NX` → `NW10 0NX`). The first match is the **pickup**, the second is the **drop-off**.
2. **Geocode** — each postcode is resolved to latitude/longitude via the free, key-less [postcodes.io](https://postcodes.io) API (async `httpx`, 5-second timeout, both lookups run concurrently).
3. **Distance** — the great-circle distance between the two points is computed with the **Haversine formula** (Earth radius 3956 mi) and rounded to 1 decimal place.

On success, `miles` is populated and `notes` gains a clear marker:

```
Miles estimated from postcodes NW10 0NX → HA9 6FF via postcodes.io
```

The `pickup_postcode` and `dropoff_postcode` fields are populated whenever postcodes are found in the screenshot — **independently** of whether the distance calculation succeeds.

### When it runs

Estimation is attempted **only** when all of the following hold:

- `platform == "deliveroo"`
- `miles is null` (i.e. the model didn't read a native distance)
- `raw_text` is non-empty
- **two** postcodes are found

Uber Eats and Deliveroo V1 (which show native miles) are never touched.

### Important notes & limitations

- **Estimated, not native.** For Deliveroo V2, `miles` is a straight-line estimate, always labelled in `notes`. It is *not* a driving distance — real road mileage will typically be somewhat higher.
- **Requires both postcodes visible.** If the screenshot crops one of the addresses, or only one postcode is found, `miles` stays `null` (the single postcode is still surfaced in `pickup_postcode`).
- **postcodes.io must be reachable.** If the API is down, returns 404 for a postcode, or exceeds the 5-second budget, estimation is skipped silently — `miles` stays `null`, `notes` is unchanged, and the request never fails. **The feature is completely invisible when it can't help.**
- **UK postcodes only.** postcodes.io covers UK postcodes exclusively, which matches the service's UK gig-economy scope.
- **Confidence is preserved.** A successful estimate does not change the `confidence` label.

---

## Tech Stack

- **FastAPI** — async HTTP framework, auto Swagger UI
- **Google Gemini 2.5 Flash Vision** — OCR + structured extraction (`max_output_tokens=1024`)
- **Pydantic v2** — request/response validation and the typed `ExtractionResult` schema
- **Pillow** — image decoding and teal-pixel platform detection
- **httpx** — async client for postcodes.io geocoding (5s timeout, fail-silent)
- **postcodes.io** — free, key-less UK postcode → lat/lng geocoding
- **HuggingFace Spaces (Docker)** — production hosting
- **pytest + pytest-asyncio** — 109-test suite, all Gemini and geocoding calls mocked

---

## Local Development

```bash
# 1. Clone
git clone <repo-url>
cd rideup-ocr-backend

# 2. Install
pip install -r requirements.txt

# 3. Configure — get a free key at https://aistudio.google.com/apikey
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=<your-key>

# 4. Run
uvicorn main:app --reload
```

Then open:

- Swagger UI → http://localhost:8000/docs
- Health → http://localhost:8000/health

### Running the tests

```bash
pytest -v
```

The suite mocks every Gemini call — no API key or network access required.

---

## Deployment (HuggingFace Spaces)

This repo is HuggingFace-ready out of the box.

1. Create a **Docker SDK** Space at https://huggingface.co/spaces.
2. Push this directory to the Space's git remote.
3. In **Settings → Variables and secrets**, add `GEMINI_API_KEY` as a **secret**.
4. HuggingFace builds the `Dockerfile`, exposes port 7860, and runs `uvicorn main:app --host 0.0.0.0 --port 7860`.
5. The `HEALTHCHECK` instruction pings `/health` every 30s.

Live deployment: https://huggingface.co/spaces/Khalid147/rideup-ocr-backend

---

## Notes for Production

- **Gemini quota** — the live demo runs on Google's **free tier** (~15 requests/min, 1,500/day). This is sufficient for trial and early production. Before scaling to real driver traffic, upgrade the project to a **paid Gemini API plan** at https://aistudio.google.com/ to lift the rate cap and remove daily quotas.
- **Secrets handling** — every credential comes from environment variables. `.env` is gitignored; only `.env.example` (with placeholders) is committed. Never commit a real `GEMINI_API_KEY`.
- **Rate-limit resilience** — the backend already absorbs Gemini 429s with 15s/30s/60s backoff and downgrades unrecoverable 502s to a low-confidence HTTP 200 response, so frontends can keep a uniform success path.
- **Monitoring** — `/health` is suitable for uptime probes; the `notes` field surfaces `retry_attempted=true` whenever a JSON-repair retry fired, which is a useful prompt-tuning signal.
