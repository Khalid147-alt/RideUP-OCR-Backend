---
title: RideUP OCR Backend
emoji: 🚗
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: UK delivery screenshot OCR via Gemini Vision
---

# RideUP OCR Backend

Production OCR API for **UK gig-economy delivery screenshots** — Uber Eats,
Deliveroo, Stuart, Just Eat, and RideUp. Upload a screenshot, get back clean
structured JSON: pay, distance, time, order count, platform, and a server-side
confidence rating.

Powered by **Google Gemini 2.5 Flash** (free tier eligible). Built with
**FastAPI** + **Pydantic v2**. Designed to deploy to **HuggingFace Spaces**
in one click.

---

## Table of contents

1. [Overview](#1-overview)
2. [Quick start (local, 3 steps)](#2-quick-start-local-3-steps)
3. [Environment variables](#3-environment-variables)
4. [Endpoints](#4-endpoints)
5. [Example response (every field explained)](#5-example-response-every-field-explained)
6. [Confidence scoring](#6-confidence-scoring)
7. [Supported platforms](#7-supported-platforms)
8. [Image format and size limits](#8-image-format-and-size-limits)
9. [HuggingFace Spaces deployment](#9-huggingface-spaces-deployment)
10. [Base44 integration guide](#10-base44-integration-guide)
11. [Testing](#11-testing)

---

## 1. Overview

`rideup-ocr-backend` accepts a single screenshot from a UK delivery driver app
and returns extracted earnings/trip data as JSON. It is purpose-built for the
visual conventions of UK delivery apps:

- Dark-mode UIs with low contrast
- `£` GBP amounts
- `X min (X mi) total` distance/time format
- Multi-stop batched orders
- Partially visible UI elements (toasts, overlays, status bars)

The server adds two guarantees on top of the raw model output:

- **Server-side confidence override** — the API never blindly trusts the
  model's self-reported confidence; confidence is recomputed from the actual
  fields present.
- **One-retry JSON repair** — if the first model response is not valid JSON,
  the server retries once with a stricter prompt before giving up.

**Why Gemini?** Google's `gemini-2.5-flash` has a genuinely free tier
(~15 requests/minute, 1,500/day) that's more than enough for early production
use. OCR quality on UK delivery screenshots is on par with paid GPT-4o.

---

## 2. Quick start (local, 3 steps)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# then edit .env and set GEMINI_API_KEY
# get a free key at: https://aistudio.google.com/apikey

# 3. Run
uvicorn main:app --reload
```

Then open:

- **Swagger UI** → http://localhost:8000/docs
- **Health** → http://localhost:8000/health

---

## 3. Environment variables

| Variable           | Required | Default              | Description                                         |
| ------------------ | -------- | -------------------- | --------------------------------------------------- |
| `GEMINI_API_KEY`   | yes      | —                    | Google Gemini API key (free at aistudio.google.com) |
| `MODEL_NAME`       | no       | `gemini-2.5-flash`   | Gemini vision model identifier                      |
| `MAX_FILE_SIZE_MB` | no       | `10`                 | Maximum upload size in megabytes                    |
| `PORT`             | no       | `7860`               | HTTP port (HuggingFace requires 7860)               |
| `ENVIRONMENT`      | no       | `production`         | Label: `production` / `staging` / `development`     |

Secrets come **exclusively** from environment variables. Nothing is ever
hardcoded. `.env` is gitignored — only `.env.example` (with placeholders)
should be committed.

---

## 4. Endpoints

### `POST /extract`

Multipart upload of an image file.

```bash
curl -X POST http://localhost:8000/extract \
  -F "image=@./screenshot.png" \
  -F "hint=uber_eats"
```

| Field   | Type           | Required | Description                                   |
| ------- | -------------- | -------- | --------------------------------------------- |
| `image` | file           | yes      | JPG, PNG, or WEBP. Max 10 MB.                 |
| `hint`  | string (form)  | no       | Optional platform hint, e.g. `deliveroo`.     |

Status codes:

- `200` — extraction succeeded
- `400` — bad input (wrong type, too large, empty, malformed)
- `422` — request body validation failed
- `500` — server misconfiguration
- `502` — upstream vision service unavailable

### `POST /extract/base64`

JSON payload with a base64-encoded image. Useful for mobile clients.

```bash
curl -X POST http://localhost:8000/extract/base64 \
  -H "Content-Type: application/json" \
  -d '{
        "image": "iVBORw0KGgoAAAANSUhEUg...",
        "hint": "deliveroo"
      }'
```

The `image` field may be a bare base64 string **or** a full
`data:image/png;base64,...` data URL — the prefix is stripped automatically.

### `GET /health`

```bash
curl http://localhost:8000/health
# → {"status":"ok","model":"gemini-2.5-flash","version":"1.0.0"}
```

### `GET /`

API metadata and endpoint index.

### `GET /docs`

Interactive Swagger UI (auto-generated by FastAPI).

---

## 5. Example response (every field explained)

```json
{
  "pay": 8.42,
  "currency": "GBP",
  "miles": 3.1,
  "minutes": 24,
  "orders": 2,
  "platform": "uber_eats",
  "confidence": "high",
  "notes": "Two-stop batch order detected.",
  "raw_text": "£8.42 · 24 min (3.1 mi) total · 2 deliveries"
}
```

| Field        | Type              | Notes                                                           |
| ------------ | ----------------- | --------------------------------------------------------------- |
| `pay`        | `float \| null`   | Monetary amount. `null` when not visible.                       |
| `currency`   | `string`          | `"GBP"`, `"USD"`, `"EUR"`, or `"unknown"`.                      |
| `miles`      | `float \| null`   | Distance in miles (km auto-converted; conversion noted).        |
| `minutes`    | `int \| null`     | Total minutes.                                                  |
| `orders`     | `int \| null`     | Number of orders/stops in the batch.                            |
| `platform`   | `string`          | One of the supported platform slugs, or `"unknown"`.            |
| `confidence` | `string`          | `"high"` / `"medium"` / `"low"`, recomputed server-side.        |
| `notes`      | `string`          | Edge cases, ambiguities, conversions applied.                   |
| `raw_text`   | `string`          | Raw text the model detected — useful for debugging.             |

Error responses use a uniform envelope:

```json
{
  "error": "bad_request",
  "detail": "Image too large: 12.30 MB exceeds limit of 10 MB.",
  "status_code": 400
}
```

---

## 6. Confidence scoring

The four **core fields** are: `pay`, `miles`, `minutes`, `orders`.

| Confidence | When                                                                |
| ---------- | ------------------------------------------------------------------- |
| `high`     | All 4 core fields extracted and model did not self-flag low/medium  |
| `medium`   | Exactly 3 of 4 core fields extracted, **or** all 4 with low self-claim |
| `low`      | Fewer than 3 core fields, or image flagged as poor quality          |

Poor-quality images (< 300×300 px, or < 10 KB) are always capped at `low` and
annotated in `notes`.

---

## 7. Supported platforms

The `platform` field can return:

- `uber_eats`
- `deliveroo`
- `stuart`
- `just_eat`
- `rideup`
- `unknown`

Detection uses visual cues (UI style, accent colours, branding) **and**
optional caller hint. The hint is treated as a suggestion — the model verifies
against the image before committing.

---

## 8. Image format and size limits

- **Allowed formats:** JPG, JPEG, PNG, WEBP
- **Max size:** 10 MB (`MAX_FILE_SIZE_MB`)
- **Min size for "good" quality:** 10 KB and 300×300 px

Anything else returns a `400` or is processed but downgraded to `low`
confidence with an explanatory note.

---

## 9. HuggingFace Spaces deployment

This repo is HuggingFace-ready out of the box.

1. **Create a Space** — at https://huggingface.co/spaces, pick the **Docker**
   SDK.
2. **Push this directory** as the Space's repository root.
3. **Add the secret** `GEMINI_API_KEY` in the Space's *Settings → Variables and
   secrets* panel. Mark it as a secret, not a variable.
4. HuggingFace will build the `Dockerfile`, expose port `7860`, and run
   `uvicorn main:app --host 0.0.0.0 --port 7860`.
5. The `HEALTHCHECK` instruction in the Dockerfile pings `/health` every 30s.

Your Space URL will look like
`https://<username>-<space-name>.hf.space`. The endpoints listed above are
available at the root of that URL.

---

## 10. Base44 integration guide

### 10.1 Configure the backend URL

In your Base44 project's environment, add:

```
RIDEUP_OCR_URL=https://<username>-<space-name>.hf.space
```

### 10.2 The exact `fetch()` call

Two flavours — pick whichever fits your client. **Both are recommended for
Base44**:

#### Multipart (file picker / camera capture)

```javascript
async function extractFromFile(file) {
  const form = new FormData();
  form.append("image", file);
  // Optional: form.append("hint", "uber_eats");

  const res = await fetch(`${import.meta.env.RIDEUP_OCR_URL}/extract`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Extraction failed (${res.status})`);
  }
  return await res.json(); // ExtractionResult
}
```

#### Base64 (mobile / canvas-captured)

```javascript
async function extractFromBase64(base64String, hint) {
  const res = await fetch(`${import.meta.env.RIDEUP_OCR_URL}/extract/base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: base64String, hint }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Extraction failed (${res.status})`);
  }
  return await res.json();
}
```

### 10.3 Handling the JSON response

```javascript
const result = await extractFromFile(file);

// Always check confidence before trusting the values.
if (result.confidence === "low") {
  showWarning("We couldn't read this screenshot clearly. Please re-upload.");
}

// Use null-checks for every numeric field.
const pay      = result.pay      ?? "—";
const miles    = result.miles    ?? "—";
const minutes  = result.minutes  ?? "—";
const orders   = result.orders   ?? "—";
const currency = result.currency === "unknown" ? "" : result.currency;
const platform = result.platform.replaceAll("_", " ");
```

### 10.4 Displaying the extracted fields in the UI

Suggested Base44 component layout:

```jsx
<TripCard>
  <Row label="Pay">       {currency} {pay}     </Row>
  <Row label="Distance">  {miles} mi           </Row>
  <Row label="Time">      {minutes} min        </Row>
  <Row label="Orders">    {orders}             </Row>
  <Row label="Platform">  {platform}           </Row>
  <ConfidenceBadge level={result.confidence} />
  {result.notes && <Note>{result.notes}</Note>}
</TripCard>
```

### 10.5 Error state handling

The backend returns a uniform error envelope:

```json
{ "error": "bad_request", "detail": "Image too large: ...", "status_code": 400 }
```

Recommended UI mapping:

| `status_code` | What happened                          | Show to the user                                |
| ------------- | -------------------------------------- | ----------------------------------------------- |
| `400`         | Bad input (size/type/empty)            | "That file can't be processed — please re-upload a JPG/PNG/WEBP under 10 MB." |
| `422`         | Malformed request body                 | "We couldn't process this request. Try again."  |
| `500`         | Server misconfigured                   | "Service unavailable. Please contact support."  |
| `502`         | Upstream vision service failure        | "Our OCR provider is temporarily down. Try again in a minute." |

---

## 11. Testing

```bash
pip install -r requirements.txt pytest pytest-asyncio
pytest -v
```

The suite mocks all Gemini calls — no API key or network access is required
to run the tests.

Covered:

- `GET /health` returns 200 with the expected shape
- Non-image uploads (e.g. PDF) are rejected with 400
- Oversize uploads are rejected with 400
- Valid uploads return an `ExtractionResult` with all fields and an
  `X-Confidence` response header
- `calculate_confidence` unit tests for all rule branches
- `POST /extract/base64` happy path, data-URL prefix tolerance, and invalid
  base64 rejection
- JSON parse retry path: first response unparseable → one retry → success

---

### Project layout

```
rideup-ocr-backend/
├── main.py              FastAPI app + all endpoints
├── models.py            Pydantic v2 input/output models
├── extractor.py         Gemini Vision extraction logic
├── prompts.py           All prompt templates
├── config.py            Settings + env vars
├── .env.example         Environment variable template
├── .gitignore
├── requirements.txt     Dependencies
├── Dockerfile           HuggingFace Spaces deployment
├── pytest.ini           Pytest configuration
├── README.md            This file
└── tests/
    ├── conftest.py
    └── test_extract.py  Test suite
```
