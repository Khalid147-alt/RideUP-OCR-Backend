/**
 * UploadTripScreenshot — Base44 React component.
 *
 * Drag-and-drop or click-to-upload area that posts a delivery screenshot to
 * the RideUp OCR Backend and renders the extracted trip data.
 *
 * Usage:
 *   <UploadTripScreenshot onExtracted={(data) => console.log(data)} />
 *
 * Accepts JPG, PNG, WEBP. Up to 10 MB. No API keys needed — the backend
 * handles Gemini auth server-side.
 */

import { useState, useRef, useCallback } from "react";
import { extractOcrData } from "./rideupOcr";

const ACCEPTED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
const MAX_BYTES = 10 * 1024 * 1024; // 10 MB — matches backend limit

const CONFIDENCE_STYLE = {
  high: { bg: "#10b981", label: "High" },
  medium: { bg: "#f59e0b", label: "Medium" },
  low: { bg: "#ef4444", label: "Low — please verify" },
};

const CURRENCY_SYMBOL = { GBP: "£", USD: "$", EUR: "€" };

const styles = {
  wrapper: {
    maxWidth: 480,
    fontFamily: "system-ui, -apple-system, sans-serif",
    color: "#111",
  },
  heading: { margin: "0 0 12px", fontSize: 18, fontWeight: 600 },
  dropzone: {
    border: "2px dashed #cbd5e1",
    borderRadius: 12,
    padding: 32,
    textAlign: "center",
    cursor: "pointer",
    transition: "border-color 0.15s, background 0.15s",
    background: "#f8fafc",
  },
  dropzoneActive: {
    borderColor: "#3b82f6",
    background: "#eff6ff",
  },
  dropzoneDisabled: {
    cursor: "not-allowed",
    opacity: 0.6,
  },
  hint: { margin: 0, fontSize: 14, color: "#475569" },
  hintSmall: { margin: "4px 0 0", fontSize: 12, color: "#94a3b8" },
  hiddenInput: { display: "none" },
  preview: {
    marginTop: 16,
    maxWidth: "100%",
    maxHeight: 240,
    borderRadius: 8,
    border: "1px solid #e2e8f0",
  },
  loading: {
    marginTop: 16,
    fontSize: 14,
    color: "#475569",
  },
  error: {
    marginTop: 16,
    padding: 12,
    borderRadius: 8,
    background: "#fef2f2",
    border: "1px solid #fecaca",
    color: "#991b1b",
    fontSize: 14,
  },
  resultCard: {
    marginTop: 16,
    padding: 16,
    borderRadius: 12,
    border: "1px solid #e2e8f0",
    background: "#ffffff",
  },
  badge: (bg) => ({
    display: "inline-block",
    padding: "3px 10px",
    borderRadius: 999,
    background: bg,
    color: "#fff",
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: 0.3,
    textTransform: "uppercase",
  }),
  table: { width: "100%", marginTop: 12, borderCollapse: "collapse" },
  rowLabel: {
    padding: "8px 0",
    color: "#64748b",
    borderBottom: "1px solid #f1f5f9",
    width: "40%",
    fontSize: 13,
  },
  rowValue: {
    padding: "8px 0",
    borderBottom: "1px solid #f1f5f9",
    fontWeight: 600,
    fontSize: 14,
  },
  payValue: {
    padding: "8px 0",
    borderBottom: "1px solid #f1f5f9",
    fontWeight: 700,
    fontSize: 18,
    color: "#0f172a",
  },
  notes: {
    marginTop: 12,
    padding: 10,
    borderRadius: 6,
    background: "#f8fafc",
    fontSize: 12,
    color: "#475569",
  },
};

function formatPay(pay, currency) {
  if (pay == null) return "N/A";
  const symbol = CURRENCY_SYMBOL[currency] ?? "";
  return `${symbol}${pay.toFixed(2)}`;
}

function formatPlatform(platform) {
  if (!platform || platform === "unknown") return "Unknown";
  return platform
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export default function UploadTripScreenshot({ onExtracted }) {
  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  const handleFile = useCallback(
    async (file) => {
      if (!file) return;

      // Client-side validation that mirrors the backend's limits.
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError("Please upload a JPG, PNG, or WEBP image.");
        setStatus("error");
        return;
      }
      if (file.size > MAX_BYTES) {
        setError("Image must be under 10 MB.");
        setStatus("error");
        return;
      }

      setPreview(URL.createObjectURL(file));
      setError(null);
      setResult(null);
      setStatus("loading");

      try {
        const data = await extractOcrData(file);
        setResult(data);
        setStatus("success");
        if (typeof onExtracted === "function") onExtracted(data);
      } catch (err) {
        setError(err?.message || "Something went wrong. Please try again.");
        setStatus("error");
      }
    },
    [onExtracted]
  );

  const onDragOver = (e) => {
    e.preventDefault();
    if (status !== "loading") setIsDragging(true);
  };
  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };
  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (status === "loading") return;
    const file = e.dataTransfer?.files?.[0];
    handleFile(file);
  };
  const onClickDropzone = () => {
    if (status !== "loading") inputRef.current?.click();
  };
  const onInputChange = (e) => {
    const file = e.target.files?.[0];
    handleFile(file);
    // Reset so the same file can be re-selected.
    e.target.value = "";
  };

  const dropzoneStyle = {
    ...styles.dropzone,
    ...(isDragging ? styles.dropzoneActive : null),
    ...(status === "loading" ? styles.dropzoneDisabled : null),
  };

  const badge = result ? CONFIDENCE_STYLE[result.confidence] : null;

  return (
    <div style={styles.wrapper}>
      <h2 style={styles.heading}>Upload trip screenshot</h2>

      <div
        style={dropzoneStyle}
        onClick={onClickDropzone}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") onClickDropzone();
        }}
      >
        <p style={styles.hint}>
          <strong>Drag &amp; drop</strong> a screenshot here, or{" "}
          <span style={{ color: "#3b82f6" }}>click to browse</span>
        </p>
        <p style={styles.hintSmall}>JPG, PNG, or WEBP · up to 10 MB</p>
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={onInputChange}
          style={styles.hiddenInput}
          disabled={status === "loading"}
        />
      </div>

      {preview && <img src={preview} alt="preview" style={styles.preview} />}

      {status === "loading" && (
        <p style={styles.loading}>Extracting trip data…</p>
      )}

      {status === "error" && <div style={styles.error}>{error}</div>}

      {status === "success" && result && (
        <div style={styles.resultCard}>
          <span style={styles.badge(badge.bg)}>{badge.label} confidence</span>

          <table style={styles.table}>
            <tbody>
              <tr>
                <td style={styles.rowLabel}>Pay</td>
                <td style={styles.payValue}>
                  {formatPay(result.pay, result.currency)}
                </td>
              </tr>
              <tr>
                <td style={styles.rowLabel}>Miles</td>
                <td style={styles.rowValue}>
                  {result.miles != null ? `${result.miles} mi` : "N/A"}
                </td>
              </tr>
              <tr>
                <td style={styles.rowLabel}>Minutes</td>
                <td style={styles.rowValue}>
                  {result.minutes != null ? `${result.minutes} min` : "N/A"}
                </td>
              </tr>
              <tr>
                <td style={styles.rowLabel}>Orders</td>
                <td style={styles.rowValue}>
                  {result.orders != null ? result.orders : "N/A"}
                </td>
              </tr>
              <tr>
                <td style={styles.rowLabel}>Platform</td>
                <td style={styles.rowValue}>
                  {formatPlatform(result.platform)}
                </td>
              </tr>
            </tbody>
          </table>

          {result.notes && (
            <div style={styles.notes}>
              <strong>Notes:</strong> {result.notes}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
