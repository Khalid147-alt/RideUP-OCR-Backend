/**
 * UploadTripScreenshot — example React component for Base44.
 *
 * Drop this in your Base44 project. Adapt the className / styling to match
 * your design system. Uses the rideupOcr client from ./lib/rideupOcr.
 */

import { useState, useRef } from "react";
import {
  extractFromFile,
  formatForDisplay,
  OcrError,
} from "./lib/rideupOcr";

const CONFIDENCE_BADGE_STYLE = {
  high: { bg: "#10b981", label: "High confidence" },
  medium: { bg: "#f59e0b", label: "Medium confidence" },
  low: { bg: "#ef4444", label: "Low confidence — please verify" },
};

export default function UploadTripScreenshot({ onExtracted }) {
  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const fileInputRef = useRef(null);

  async function handleFile(file) {
    if (!file) return;

    // Client-side guardrails matching the backend's limits.
    if (file.size > 10 * 1024 * 1024) {
      setError("Image must be under 10 MB.");
      setStatus("error");
      return;
    }
    if (!/^image\/(jpe?g|png|webp)$/i.test(file.type)) {
      setError("Please upload a JPG, PNG, or WEBP image.");
      setStatus("error");
      return;
    }

    setPreview(URL.createObjectURL(file));
    setStatus("loading");
    setError(null);

    try {
      const extracted = await extractFromFile(file);
      setResult(extracted);
      setStatus("success");
      onExtracted?.(extracted);
    } catch (err) {
      const msg =
        err instanceof OcrError
          ? err.message
          : "Something went wrong. Please try again.";
      setError(msg);
      setStatus("error");
    }
  }

  const display = result ? formatForDisplay(result) : null;
  const badge = result ? CONFIDENCE_BADGE_STYLE[result.confidence] : null;

  return (
    <div style={{ maxWidth: 480, fontFamily: "system-ui, sans-serif" }}>
      <h2>Upload trip screenshot</h2>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={(e) => handleFile(e.target.files?.[0])}
        disabled={status === "loading"}
      />

      {preview && (
        <div style={{ marginTop: 16 }}>
          <img
            src={preview}
            alt="preview"
            style={{
              maxWidth: "100%",
              maxHeight: 240,
              borderRadius: 8,
              border: "1px solid #ddd",
            }}
          />
        </div>
      )}

      {status === "loading" && (
        <p style={{ marginTop: 16 }}>Extracting trip data…</p>
      )}

      {status === "error" && (
        <div
          style={{
            marginTop: 16,
            padding: 12,
            background: "#fee2e2",
            border: "1px solid #fecaca",
            borderRadius: 8,
            color: "#991b1b",
          }}
        >
          {error}
        </div>
      )}

      {status === "success" && display && (
        <div style={{ marginTop: 16 }}>
          <span
            style={{
              display: "inline-block",
              padding: "4px 10px",
              borderRadius: 999,
              background: badge.bg,
              color: "white",
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            {badge.label}
          </span>

          <table
            style={{
              width: "100%",
              marginTop: 12,
              borderCollapse: "collapse",
            }}
          >
            <tbody>
              <Row label="Pay" value={display.pay} />
              <Row label="Distance" value={display.miles} />
              <Row label="Time" value={display.minutes} />
              <Row label="Orders" value={display.orders} />
              <Row label="Platform" value={display.platform} />
            </tbody>
          </table>

          {display.notes && (
            <p style={{ marginTop: 12, fontSize: 13, color: "#666" }}>
              <strong>Notes:</strong> {display.notes}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function Row({ label, value }) {
  return (
    <tr>
      <td
        style={{
          padding: "8px 0",
          color: "#666",
          borderBottom: "1px solid #eee",
          width: "40%",
        }}
      >
        {label}
      </td>
      <td
        style={{
          padding: "8px 0",
          borderBottom: "1px solid #eee",
          fontWeight: 600,
        }}
      >
        {value}
      </td>
    </tr>
  );
}
