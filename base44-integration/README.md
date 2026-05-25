# Base44 integration files

Two drop-in files for wiring the RideUP OCR backend into your Base44 frontend.

## Files

| File | Where to put it in Base44 |
|------|---------------------------|
| `rideupOcr.js` | `src/lib/rideupOcr.js` |
| `UploadTripScreenshot.jsx` | `src/components/UploadTripScreenshot.jsx` |

## Setup (3 steps)

### 1. Configure the backend URL

Add to your Base44 project's `.env` (or environment settings):

```
VITE_RIDEUP_OCR_URL=https://YOUR-USERNAME-rideup-ocr-backend.hf.space
```

(Use `NEXT_PUBLIC_RIDEUP_OCR_URL` if Base44 uses Next.js conventions instead of Vite.)

### 2. Import and use the component

```jsx
import UploadTripScreenshot from "./components/UploadTripScreenshot";

function TripEntryPage() {
  return (
    <UploadTripScreenshot
      onExtracted={(result) => {
        // result is the parsed ExtractionResult
        // Save to your DB, populate a form, etc.
        console.log("Extracted:", result);
      }}
    />
  );
}
```

### 3. Or use the client directly without the example UI

```jsx
import { extractFromFile, extractFromBase64, formatForDisplay } from "./lib/rideupOcr";

const result = await extractFromFile(file);
const display = formatForDisplay(result);
// display.pay = "£8.42", display.miles = "3.1 mi", etc.
```

## API surface of `rideupOcr.js`

| Export | Purpose |
|--------|---------|
| `extractFromFile(file, hint?)` | Multipart upload of a File/Blob |
| `extractFromBase64(b64, hint?)` | JSON POST with base64 string (accepts data URLs too) |
| `checkHealth()` | Ping `/health` — useful at app startup |
| `formatForDisplay(result)` | Turn nullable fields into UI-ready strings (`"£8.42"`, `"3.1 mi"`, `"—"`) |
| `OcrError` | Custom error class — has `.status` and `.detail` |

## Error handling

Every function throws `OcrError` on failure. The shape lets you map HTTP
status codes to user-friendly messages:

```js
try {
  const result = await extractFromFile(file);
} catch (err) {
  if (err instanceof OcrError) {
    switch (err.status) {
      case 400: setMessage("That file can't be processed."); break;
      case 502: setMessage("OCR service is down — try again in a minute."); break;
      default:  setMessage("Something went wrong.");
    }
  }
}
```

## CORS

The backend has `allow_origins=["*"]` by default — your Base44 app will work
from any origin without extra config. If you later want to lock it down, edit
the CORS middleware in `main.py` on the backend.
