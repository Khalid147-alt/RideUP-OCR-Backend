/**
 * RideUp OCR — Base44 client.
 *
 * Lightweight wrapper around the RideUp OCR Backend. No API keys required —
 * the backend handles Gemini auth server-side. Just import and call.
 *
 * Drop this in your Base44 project (e.g. src/lib/rideupOcr.js) and import:
 *
 *   import { extractOcrData, extractOcrDataBase64 } from "./lib/rideupOcr";
 *
 *   const result = await extractOcrData(file);
 *   // result = { pay, currency, miles, minutes, orders, platform, confidence, notes, raw_text }
 */

const RIDEUP_OCR_URL = "https://khalid147-rideup-ocr-backend.hf.space";

/**
 * @typedef {Object} ExtractionResult
 * @property {number|null} pay        Trip earnings (e.g. 11.08). Null if unreadable.
 * @property {"GBP"|"USD"|"EUR"|"unknown"} currency
 * @property {number|null} miles      Distance in miles. Null for Deliveroo V2 (expected).
 * @property {number|null} minutes    Duration in minutes. Null for Deliveroo V2 (expected).
 * @property {number|null} orders     Number of orders / stops.
 * @property {"uber_eats"|"deliveroo"|"unknown"} platform
 * @property {"high"|"medium"|"low"} confidence
 * @property {string} notes           Layout detected, edge cases, retry markers.
 * @property {string} raw_text        Raw OCR text — useful for debugging.
 */

/**
 * Extract structured trip data from an image File / Blob.
 *
 * Posts the image as multipart/form-data to `POST /extract`.
 *
 * @param {File|Blob} imageFile  An image File from an `<input type="file">`
 *                               or a Blob from a canvas / camera capture.
 *                               Must be JPG, PNG, or WEBP, under 10 MB.
 * @returns {Promise<ExtractionResult>} Structured trip data.
 * @throws {Error} If the request fails or the backend returns a non-2xx status.
 */
export async function extractOcrData(imageFile) {
  if (!imageFile) {
    throw new Error("extractOcrData: no image file provided");
  }

  const formData = new FormData();
  formData.append("image", imageFile);

  try {
    const response = await fetch(`${RIDEUP_OCR_URL}/extract`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let detail = `OCR request failed (${response.status})`;
      try {
        const errBody = await response.json();
        if (errBody?.detail) detail = errBody.detail;
      } catch {
        /* response body was not JSON — keep generic detail */
      }
      throw new Error(detail);
    }

    return await response.json();
  } catch (err) {
    // Re-throw with a stable message so the caller can show it to the user.
    if (err instanceof Error) throw err;
    throw new Error("OCR request failed: unknown error");
  }
}

/**
 * Extract structured trip data from a base64-encoded image string.
 *
 * Posts JSON to `POST /extract/base64`. Accepts either a bare base64 string
 * or a full data URL (e.g. "data:image/png;base64,iVBORw0K..."); the backend
 * strips the prefix automatically.
 *
 * @param {string} base64String  Base64 image data, with or without a data: prefix.
 * @returns {Promise<ExtractionResult>} Structured trip data.
 * @throws {Error} If the request fails or the backend returns a non-2xx status.
 */
export async function extractOcrDataBase64(base64String) {
  if (!base64String) {
    throw new Error("extractOcrDataBase64: no image data provided");
  }

  try {
    const response = await fetch(`${RIDEUP_OCR_URL}/extract/base64`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64String }),
    });

    if (!response.ok) {
      let detail = `OCR request failed (${response.status})`;
      try {
        const errBody = await response.json();
        if (errBody?.detail) detail = errBody.detail;
      } catch {
        /* response body was not JSON — keep generic detail */
      }
      throw new Error(detail);
    }

    return await response.json();
  } catch (err) {
    if (err instanceof Error) throw err;
    throw new Error("OCR request failed: unknown error");
  }
}
