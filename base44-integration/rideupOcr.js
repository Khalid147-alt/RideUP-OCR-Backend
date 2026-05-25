/**
 * RideUP OCR client — minimal wrapper around the backend.
 *
 * Drop this in your Base44 project at: src/lib/rideupOcr.js
 *
 * Usage:
 *   import { extractFromFile, extractFromBase64, checkHealth } from "./lib/rideupOcr";
 *
 *   const result = await extractFromFile(file);
 *   if (result.confidence === "low") { ... }
 */

// Configure your backend URL via env. Examples:
//   Vite:        import.meta.env.VITE_RIDEUP_OCR_URL
//   CRA / Next:  process.env.NEXT_PUBLIC_RIDEUP_OCR_URL
//   Or hardcode the HuggingFace URL below as a fallback.
const BACKEND_URL =
  (typeof import.meta !== "undefined" &&
    import.meta.env &&
    (import.meta.env.VITE_RIDEUP_OCR_URL ||
      import.meta.env.RIDEUP_OCR_URL)) ||
  (typeof process !== "undefined" &&
    process.env &&
    process.env.NEXT_PUBLIC_RIDEUP_OCR_URL) ||
  "https://YOUR-USERNAME-rideup-ocr-backend.hf.space";

class OcrError extends Error {
  constructor(message, { status, detail } = {}) {
    super(message);
    this.name = "OcrError";
    this.status = status;
    this.detail = detail;
  }
}

async function _parseError(res) {
  let body = {};
  try {
    body = await res.json();
  } catch {
    /* ignore */
  }
  return new OcrError(body.detail || `Extraction failed (${res.status})`, {
    status: res.status,
    detail: body.detail,
  });
}

/**
 * Extract from a File / Blob (typical file-picker or camera capture).
 * @param {File|Blob} file
 * @param {string} [hint] - Optional platform hint, e.g. "uber_eats".
 * @returns {Promise<ExtractionResult>}
 */
export async function extractFromFile(file, hint) {
  if (!file) throw new OcrError("No file provided");
  const form = new FormData();
  form.append("image", file);
  if (hint) form.append("hint", hint);

  const res = await fetch(`${BACKEND_URL}/extract`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw await _parseError(res);
  return res.json();
}

/**
 * Extract from a base64 string (mobile / canvas-captured / pasted data URL).
 * Accepts both bare base64 and "data:image/png;base64,..." URLs.
 * @param {string} base64
 * @param {string} [hint]
 * @returns {Promise<ExtractionResult>}
 */
export async function extractFromBase64(base64, hint) {
  if (!base64) throw new OcrError("No image data provided");
  const res = await fetch(`${BACKEND_URL}/extract/base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: base64, hint }),
  });
  if (!res.ok) throw await _parseError(res);
  return res.json();
}

/**
 * Liveness probe.
 * @returns {Promise<{status: string, model: string, version: string}>}
 */
export async function checkHealth() {
  const res = await fetch(`${BACKEND_URL}/health`);
  if (!res.ok) throw new OcrError("Health check failed", { status: res.status });
  return res.json();
}

/**
 * Format an ExtractionResult for UI display, handling null fields cleanly.
 * @param {ExtractionResult} result
 */
export function formatForDisplay(result) {
  const dash = "—";
  return {
    pay:
      result.pay != null
        ? `${result.currency === "GBP" ? "£" : result.currency === "USD" ? "$" : result.currency === "EUR" ? "€" : ""}${result.pay.toFixed(2)}`
        : dash,
    miles: result.miles != null ? `${result.miles} mi` : dash,
    minutes: result.minutes != null ? `${result.minutes} min` : dash,
    orders: result.orders != null ? String(result.orders) : dash,
    platform: result.platform.replace(/_/g, " "),
    confidence: result.confidence,
    notes: result.notes || "",
  };
}

export { OcrError };

/**
 * @typedef {Object} ExtractionResult
 * @property {number|null} pay
 * @property {"GBP"|"USD"|"EUR"|"unknown"} currency
 * @property {number|null} miles
 * @property {number|null} minutes
 * @property {number|null} orders
 * @property {"uber_eats"|"deliveroo"|"stuart"|"just_eat"|"rideup"|"unknown"} platform
 * @property {"high"|"medium"|"low"} confidence
 * @property {string} notes
 * @property {string} raw_text
 */
