"""Prompt templates for Gemini Vision extraction.

The master prompt is tuned for UK gig-economy delivery screenshots — dark mode
UIs, GBP £ amounts, batched multi-stop offers, partially visible elements, and
low-contrast green/white-on-black colour schemes.
"""

from __future__ import annotations

# Strict JSON schema reproduced verbatim in the prompt so GPT-4o cannot drift.
EXPECTED_SCHEMA_EXAMPLE = """{
  "pay": 8.42,
  "currency": "GBP",
  "miles": 3.1,
  "minutes": 24,
  "orders": 2,
  "platform": "uber_eats",
  "confidence": "high",
  "notes": "Two-stop batch order detected in offer card.",
  "raw_text": "£8.42 · 24 min (3.1 mi) total · 2 deliveries"
}"""

MASTER_EXTRACTION_PROMPT = f"""You are an expert data extractor for UK gig-economy delivery driver screenshots.
You receive a single screenshot from one of the following apps:
- Uber Eats (driver/courier app — black UI, green accents, £ amounts)
- Deliveroo (riders app — teal/cyan accents on dark background)
- Stuart (couriers app — yellow/black branding)
- Just Eat (couriers — orange/red accents)
- RideUp (custom UK driver platform)

Your job is to extract structured trip/offer information and return it as ONE
valid JSON object — nothing else. No prose, no markdown fences, no preamble.

# Fields to extract
- pay           : float  — monetary payment offered/earned (e.g. 8.42)
- currency      : string — "GBP", "USD", "EUR" or "unknown". Default to "GBP"
                  for any £ symbol.
- miles         : float  — total trip distance in miles. If shown in km,
                  convert (1 km = 0.621371 mi) and mention in notes.
- minutes       : int    — total estimated time in whole minutes.
- orders        : int    — number of orders / stops / deliveries in the batch.
                  A single delivery = 1. Look for "X deliveries", stop pins on
                  the map, or numbered stop cards.
- platform      : string — one of "uber_eats" | "deliveroo" | "stuart" |
                  "just_eat" | "rideup" | "unknown". Detect from UI style,
                  colours, fonts, and any visible branding.
- confidence    : string — "high" | "medium" | "low" (server may override).
- notes         : string — anything ambiguous, partial, unusual, or converted.
- raw_text      : string — every legible piece of text you can read, joined
                  with " · " separators. This is your evidence trail.

# Critical rules
1. Return ONE JSON object only. No markdown. No ```json fences. No text before
   or after the JSON.
2. If a field is not visible or you are not sure, set it to null. Never guess
   numeric values. Confidence and platform may use "unknown" instead of null
   because they are non-null strings.
3. UK format hints:
   - "£8.42" → pay 8.42, currency "GBP"
   - "24 min (3.1 mi) total" → minutes 24, miles 3.1
   - "2 deliveries" / "2 orders" / "Batch of 2" → orders 2
   - A single order with no batch indicator → orders 1 (only if clearly a
     single-order screen; otherwise null).
4. Detect platform from visual cues, not just text. Uber Eats uses a bold
   black UI with mint-green CTA. Deliveroo uses teal. Stuart uses yellow.
5. Self-validate before returning: are all numbers plausible? Is the JSON
   syntactically valid? Are the field types correct (float vs int)?
6. Put any uncertainty or edge-case observations into `notes`. Examples:
   "Distance shown in km, converted." or "Pay partially obscured by toast."

# One-shot example output (exactly this shape — values illustrative only)
{EXPECTED_SCHEMA_EXAMPLE}

Now read the attached screenshot and emit the JSON object.
Return ONLY the JSON object — no markdown, no fences, no commentary.
"""


def build_extraction_prompt(hint: str | None = None) -> str:
    """Return the master extraction prompt, optionally augmented with a hint.

    Args:
        hint: Optional caller-supplied platform hint, e.g. ``"uber_eats"``.

    Returns:
        The full prompt string to send as the user message to GPT-4o.
    """
    if not hint:
        return MASTER_EXTRACTION_PROMPT
    hint_clean = hint.strip().lower()[:64]
    return (
        f"{MASTER_EXTRACTION_PROMPT}\n\n"
        f"# Caller hint\nThe caller suggests this screenshot is from: "
        f"'{hint_clean}'. Treat this as a hint only — verify against the "
        f"actual visual evidence before setting `platform`."
    )


RETRY_STRICT_PROMPT = (
    "Your previous response could not be parsed as JSON. "
    "Return ONLY a single valid JSON object matching exactly this schema, "
    "with no markdown, no fences, no commentary:\n\n"
    f"{EXPECTED_SCHEMA_EXAMPLE}\n\n"
    "If a field is unknown, use null (or \"unknown\" for currency/platform). "
    "Emit the JSON now."
)
