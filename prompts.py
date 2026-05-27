"""Prompt templates for Gemini Vision extraction.

The master prompt is tuned for UK gig-economy delivery screenshots — dark mode
UIs, GBP £ amounts, batched multi-stop offers, partially visible elements, and
low-contrast green/white-on-black colour schemes.

Platform-specific prompt variants exist for the trickiest UIs (Deliveroo)
where the master prompt alone misclassifies fields. Routing happens in
``extractor._select_prompt``.
"""

from __future__ import annotations

# Strict JSON schema reproduced verbatim in the prompt so the model cannot drift.
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


# ---------------------------------------------------------------------------
# Deliveroo-specific prompt
# ---------------------------------------------------------------------------
#
# Deliveroo's UI has several quirks the master prompt doesn't always handle:
#
# - Teal/cyan accents (not green like Uber). Pay buttons and earnings banners
#   are typically on a teal/turquoise background or with teal text.
# - Pay format: often "£X.XX" but sometimes "£X.XX (Includes £Y.YY tip)" —
#   we want the gross figure (the larger one shown to the rider) into ``pay``.
# - "Single" vs "Stacked" orders:
#     * Single  → one pickup card, one drop card, often "1 delivery" implied.
#     * Stacked → two or more pickup/drop cards with a "Multi-order" or
#                 "Stacked" badge near the top.
# - Distance/time appear as "X mi · X min" or as two separate chips rather
#   than the "X min (X mi) total" Uber phrasing.
# - "Standard / Plus / Priority" labels are NOT order counts — ignore them.

DELIVEROO_EXTRACTION_PROMPT = f"""You are an expert data extractor for **Deliveroo** rider/courier screenshots in the UK.
The screenshot is from the Deliveroo Rider app. Read it and emit ONE valid
JSON object — no markdown, no fences, no prose.

# Deliveroo visual cues you can rely on
- Teal / cyan / turquoise accents (the Deliveroo brand colour).
- Dark navy or black background, white text.
- Rounded rider/restaurant cards.
- A bold £ amount near the top of an offer or earnings panel.
- Pickup pin + drop pin icons; map snippet often present.

# Deliveroo-specific extraction rules
1. **Pay**:
   - The headline £ amount is the gross pay. Use that for ``pay``.
   - If a breakdown like "£6.20 (Includes £1.00 tip)" is shown, still use the
     gross headline figure (£6.20) and mention the tip in ``notes``.
   - If only a fee + tip are listed separately, sum them.

2. **Orders (single vs stacked)**:
   - Look for words like "Stacked", "Multi-order", "2 orders", or two
     separate restaurant cards stacked vertically → orders >= 2.
   - "Standard", "Plus", "Priority", "Rider+" are SUBSCRIPTION/TIER labels,
     NOT order counts. Ignore them when counting.
   - One restaurant card + one customer drop = single delivery → orders = 1.
   - If you genuinely cannot tell, set orders to null.

3. **Distance/time format**:
   - Common Deliveroo layouts:
     "2.1 mi · 18 min"   → miles 2.1, minutes 18
     "18 min  ·  2.1 mi" → same
     Two separate chips, one with distance, one with time → combine.
   - If distance is in km ("3.4 km"), convert to miles (1 km = 0.621371 mi)
     and mention the conversion in ``notes``.

4. **Currency**: "£" → "GBP". Always.

5. **Platform**: this screenshot IS Deliveroo. Set ``platform`` = "deliveroo".

# Output schema (exact shape, no extra keys)
{EXPECTED_SCHEMA_EXAMPLE.replace('"uber_eats"', '"deliveroo"')}

Set any field you cannot read clearly to null (use "unknown" only for
``currency`` and ``platform``). Put ambiguities in ``notes``.

Emit the JSON object now. ONLY the JSON. No fences. No commentary.
"""


def build_extraction_prompt(hint: str | None = None) -> str:
    """Return the master extraction prompt, optionally augmented with a hint.

    For ``hint == "deliveroo"`` the dedicated Deliveroo prompt is returned
    instead of the master prompt — it handles the platform's quirks better.

    Args:
        hint: Optional caller-supplied platform hint, e.g. ``"uber_eats"`` or
            ``"deliveroo"``.

    Returns:
        The full prompt string to send as the user message to the vision model.
    """
    hint_clean = (hint or "").strip().lower()[:64]

    if hint_clean == "deliveroo":
        return DELIVEROO_EXTRACTION_PROMPT

    if not hint_clean:
        return MASTER_EXTRACTION_PROMPT

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
