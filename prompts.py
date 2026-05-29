"""Prompt templates for Gemini Vision extraction.

The master prompt is tuned for UK gig-economy delivery screenshots — dark mode
UIs, GBP £ amounts, batched multi-stop offers, partially visible elements, and
low-contrast green/white-on-black colour schemes.

Platform-specific prompt variants exist for the trickiest UIs (Deliveroo)
where the master prompt alone misclassifies fields. Routing happens in
``extractor._select_prompt``.

Every prompt starts with the same ``CRITICAL_OUTPUT_RULES`` block. It exists
because real-world stress testing surfaced a recurring failure mode where the
model echoed its own JSON output back into the ``raw_text`` field (e.g.
``"raw_text": "{\\"pay\\": 11"``). The block draws a hard line between "the
JSON envelope you must emit" and "the text you can literally see in the
image" so the model never confuses one for the other.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared header: separates JSON-envelope rules from per-field instructions.
# ---------------------------------------------------------------------------
#
# This block is intentionally short, blunt, and reproduced verbatim across
# every prompt template. Variations in wording invite drift, so the same
# words appear in master, Deliveroo V1, and Deliveroo V2 prompts.

CRITICAL_OUTPUT_RULES = """CRITICAL OUTPUT RULES — READ FIRST:
1. Your response must be a single raw JSON object
2. Start with { and end with } — nothing else
3. No markdown, no backticks, no "json" prefix
4. No explanation before or after the JSON

CRITICAL raw_text RULE:
raw_text must contain ONLY text that is literally
visible as words/numbers in the screenshot image.
raw_text must NEVER contain JSON syntax.
raw_text must NEVER start with { or contain \"
raw_text must NEVER echo your own JSON output.

CORRECT raw_text examples:
✓ "£11.08 · 3 orders · The Poke Shack"
✓ "£5.50 · 22 min (3.2 mi) total · Delivery (2)"
✓ "£6.39 · 1 order · Gopuff · Accept and go"

WRONG raw_text examples (never do this):
✗ "{\\"pay\\": 11}"
✗ "{\\n  \\"pay\\": 7.22"
✗ Any string starting with {

# ================ END OUTPUT RULES — FIELD DEFINITIONS FOLLOW ================
"""


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

MASTER_EXTRACTION_PROMPT = f"""{CRITICAL_OUTPUT_RULES}

You are an expert data extractor for UK gig-economy delivery driver screenshots.
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
                  with " · " separators. This is your evidence trail. It must
                  contain ONLY text visible in the image — see the output
                  rules at the top of this prompt.

# Critical field rules
1. If a field is not visible or you are not sure, set it to null. Never guess
   numeric values. Confidence and platform may use "unknown" instead of null
   because they are non-null strings.
2. UK format hints:
   - "£8.42" → pay 8.42, currency "GBP"
   - "24 min (3.1 mi) total" → minutes 24, miles 3.1
   - "38 min (7.5 mi) total" → minutes 38, miles 7.5
   - "38 min (7.5 mi)" without "total" → same: minutes 38, miles 7.5
   - "2 deliveries" / "2 orders" / "Batch of 2" → orders 2
   - "Delivery" (no number, no brackets) → orders 1
   - "Delivery (1)" → orders 1
   - "Delivery (2)" → orders 2
   - A single restaurant + single drop address → orders 1
3. Detect platform from visual cues, not just text. Uber Eats uses a bold
   black UI with mint-green CTA and a dark grey/black map. The button often
   reads "Confirm". Deliveroo uses teal accents. Stuart uses yellow.
4. Self-validate before returning: are all numbers plausible? Is the JSON
   syntactically valid? Are the field types correct (float vs int)?
5. Put any uncertainty or edge-case observations into `notes`. Examples:
   "Distance shown in km, converted." or "Pay partially obscured by toast."

# One-shot example output (exactly this shape — values illustrative only)
{EXPECTED_SCHEMA_EXAMPLE}

# One-shot example: Uber Eats SINGLE order with distance and time
{{"pay": 9.35, "currency": "GBP", "miles": 7.5, "minutes": 38, "orders": 1, "platform": "uber_eats", "confidence": "high", "notes": "Single Uber Eats delivery; green Confirm button, dark map.", "raw_text": "£9.35 · 38 min (7.5 mi) total · Delivery · patisserie land ltd · London SW1X 7JW · Confirm"}}

Now read the attached screenshot and emit the JSON object.
Return ONLY the JSON object — no markdown, no fences, no commentary.
"""


# ---------------------------------------------------------------------------
# Deliveroo V1 prompt — classic layout (distance + time visible in offer card)
# ---------------------------------------------------------------------------
#
# Deliveroo's classic UI quirks the master prompt doesn't always handle:
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

DELIVEROO_EXTRACTION_PROMPT = f"""{CRITICAL_OUTPUT_RULES}

You are an expert data extractor for **Deliveroo** rider/courier screenshots in the UK.
The screenshot is from the Deliveroo Rider app. Read it and emit ONE valid
JSON object — no markdown, no fences, no prose.

# VALUE FORMAT RULES — STRICT
- ``pay`` MUST be a JSON number, NEVER a string. WRONG: "pay": "£6.50".
  RIGHT: "pay": 6.50. Strip the £ sign yourself before emitting.
- ``miles`` MUST be a JSON number, never include "mi". WRONG: "miles": "2.1 mi".
  RIGHT: "miles": 2.1.
- ``minutes`` MUST be a JSON integer, never include "min". WRONG:
  "minutes": "18 min". RIGHT: "minutes": 18.
- ``orders`` MUST be a JSON integer. WRONG: "orders": "2 stacked". RIGHT:
  "orders": 2.
- Do NOT nest objects inside numeric fields. WRONG:
  "pay": {{"gross": 6.50, "tip": 1.00}}. RIGHT: "pay": 6.50 and mention the tip
  in ``notes``.

# FALLBACK RULE — VERY IMPORTANT
If you cannot read a field clearly, return ``null`` for that field. NEVER skip
the field entirely. NEVER return partial JSON. Every key in the schema below
MUST appear in your output, even if its value is null or "unknown".

For ``currency`` and ``platform`` use the string ``"unknown"`` instead of null
(those two fields are strings, not nullable numbers).

# Deliveroo visual cues you can rely on
- Teal / cyan / turquoise accents (the Deliveroo brand colour).
- Dark navy or black background, white text.
- Rounded rider/restaurant cards.
- A bold £ amount near the top of an offer or earnings panel.
- Pickup pin + drop pin icons; map snippet often present.

# Deliveroo-specific extraction rules
1. **Pay**:
   - The headline £ amount is the gross pay. Use that number for ``pay``.
   - If a breakdown like "£6.20 (Includes £1.00 tip)" is shown, still use the
     gross headline figure (6.20) and mention the tip in ``notes``.
   - If only a fee + tip are listed separately, sum them into one number.
   - Always strip the £ sign — emit the bare number (e.g. 6.50, not "£6.50").

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
   - Emit numbers only — no "mi" suffix, no "min" suffix.

4. **Currency**: "£" → "GBP". Always.

5. **Platform**: this screenshot IS Deliveroo. Set ``platform`` = "deliveroo".

# Output schema (exact shape, every key required, no extras)
{EXPECTED_SCHEMA_EXAMPLE.replace('"uber_eats"', '"deliveroo"')}

# One-shot example — EXACT format of a valid response
{{"pay": 6.50, "currency": "GBP", "miles": 2.1, "minutes": 18, "orders": 1, "platform": "deliveroo", "confidence": "high", "notes": "Single order, tip £1.00 included in gross pay.", "raw_text": "Deliveroo · £6.50 · 2.1 mi · 18 min · 1 delivery"}}

# Another one-shot example — stacked order with km converted
{{"pay": 11.20, "currency": "GBP", "miles": 3.7, "minutes": 32, "orders": 2, "platform": "deliveroo", "confidence": "high", "notes": "Stacked order: 2 restaurants. Distance shown as 6.0 km, converted to 3.7 mi.", "raw_text": "Stacked · £11.20 · 6.0 km · 32 min · 2 orders"}}

# Final reminder before you respond
- Start with `{{`
- End with `}}`
- All 9 keys present (pay, currency, miles, minutes, orders, platform, confidence, notes, raw_text)
- Numbers as numbers (not strings)
- raw_text contains only image text — never your own JSON

Emit the JSON object for the attached screenshot now.
"""


# ---------------------------------------------------------------------------
# Deliveroo V2 prompt — newer layout (NO distance, NO time visible)
# ---------------------------------------------------------------------------
#
# Real client stress-test screenshots revealed a second Deliveroo layout
# (we call it "V2"). It is visually distinct and structurally simpler:
#
#   - Large £XX.XX value at top left of the offer card
#   - "X orders" text with a suitcase icon at top right
#   - "Accept and go" teal button at the bottom (sometimes teal+red split)
#   - Purple/pink map with the Google Maps logo
#   - Restaurant rows show "1x" / "2x" quantity badges
#   - A "Delivered / Earned / Rejected" stats bar above the card
#   - The toast "You've been assigned a new order" sometimes appears at top
#   - Distance and minutes are NOT in the offer card AT ALL
#   - No "total" suffix anywhere
#
# Because miles and minutes are genuinely absent (not unreadable), the prompt
# must tell the model to emit ``null`` for them confidently and explain in
# ``notes`` that the data is not in the layout. Server-side confidence
# scoring (see ``extractor.calculate_confidence``) treats this as a high
# confidence outcome when pay and orders are present.

DELIVEROO_V2_EXTRACTION_PROMPT = f"""{CRITICAL_OUTPUT_RULES}

You are an expert data extractor for **Deliveroo V2** offer-card screenshots.
The Deliveroo V2 layout is the newer offer card with these visual fingerprints:

  - Large £XX.XX value at the TOP LEFT of the offer card.
  - "X orders" text at the TOP RIGHT, often next to a suitcase icon \U0001f9f3.
  - Restaurant rows with "1x" / "2x" quantity badges before the restaurant name.
  - "Accept and go" TEAL button at the bottom (sometimes a teal + red split
    button — the teal half is "Accept", the red half is "Decline").
  - A purple/pink map snippet with the Google Maps logo above the card.
  - A "Delivered / Earned / Rejected" stats bar across the top of the screen.
  - Sometimes a toast notification "You've been assigned a new order".
  - NO distance and NO minutes are displayed anywhere in the offer card.
  - NO "total" suffix anywhere on the screen.

# VALUE FORMAT RULES — STRICT
- ``pay``     MUST be a JSON number (e.g. 11.08), never a string. Strip "£".
- ``orders``  MUST be a JSON integer extracted from "X orders" — just the number.
- ``miles``   MUST be ``null`` for this layout. The data is not present.
- ``minutes`` MUST be ``null`` for this layout. The data is not present.
- ``currency`` is always "GBP" for Deliveroo V2 (UK only).
- ``platform`` is always "deliveroo".
- ``confidence`` should be "medium" when miles/minutes are null because the
  data is genuinely ABSENT from the layout (the server will lift this to
  "high" because absent != unreadable for this template).
- ``notes`` MUST include: "Deliveroo V2 layout — distance and estimated
  time not displayed in order card".

# raw_text rules for Deliveroo V2
Concatenate every legible piece of text from the BOTTOM OFFER CARD ONLY,
separated by " · ". Do NOT include the stats bar at the top. Do NOT include
your own JSON. The raw_text must contain only what is literally printed on
the offer card itself.

# Real-world examples from the client stress test

Example 1 — three-order batch:
  Pay: £11.08, Orders: "3 orders" with suitcase icon
  Restaurants: 2x The Poke Shack (255 West End Lane NW61XN),
               1x Banana Tree (237-239 West End Lane NW61XN)
  Button: teal "Accept and go"
  Expected JSON:
  {{"pay": 11.08, "currency": "GBP", "miles": null, "minutes": null, "orders": 3, "platform": "deliveroo", "confidence": "medium", "notes": "Deliveroo V2 layout — distance and estimated time not displayed in order card.", "raw_text": "£11.08 · 3 orders · 2x The Poke Shack · 255 West End Lane NW61XN · 1x Banana Tree · 237-239 West End Lane NW61XN · Accept and go"}}

Example 2 — single order:
  Pay: £6.39, Orders: "1 order"
  Pickup: Gopuff, Unit 5 Wembley Trade Park NW100JF
  Drop: 237 Willesden Lane Flat 3 NW25RT
  Button: teal+red split "Accept and go"
  Expected JSON:
  {{"pay": 6.39, "currency": "GBP", "miles": null, "minutes": null, "orders": 1, "platform": "deliveroo", "confidence": "medium", "notes": "Deliveroo V2 layout — distance and estimated time not displayed in order card.", "raw_text": "£6.39 · 1 order · Gopuff · Unit 5 Wembley Trade Park NW100JF · 237 Willesden Lane Flat 3 NW25RT · Accept and go"}}

Example 3 — two-order batch:
  Pay: £7.22, Orders: "2 orders"
  Restaurants: 1x Flùr Flowers (24B Windsor Road NW25DS),
               1x Pizza Hut Delivery (7 Walm Lane NW25SJ)
  Drop: 17 Glenbrook Road NW61TN
  Button: teal "Accept and go"
  Expected JSON:
  {{"pay": 7.22, "currency": "GBP", "miles": null, "minutes": null, "orders": 2, "platform": "deliveroo", "confidence": "medium", "notes": "Deliveroo V2 layout — distance and estimated time not displayed in order card.", "raw_text": "£7.22 · 2 orders · 1x Flùr Flowers · 24B Windsor Road NW25DS · 1x Pizza Hut Delivery · 7 Walm Lane NW25SJ · 17 Glenbrook Road NW61TN · Accept and go"}}

# Output schema (exact shape, every key required, no extras)
{EXPECTED_SCHEMA_EXAMPLE.replace('"uber_eats"', '"deliveroo"')}

# Final reminder before you respond
- Start with `{{`
- End with `}}`
- All 9 keys present
- miles and minutes are null (data not in layout)
- raw_text is the offer-card text only — NEVER your own JSON

Emit the JSON object for the attached screenshot now.
"""


def build_extraction_prompt(hint: str | None = None) -> str:
    """Return the master extraction prompt, optionally augmented with a hint.

    For ``hint == "deliveroo"`` the dedicated Deliveroo V2 prompt is returned
    — V2 is the layout that the real client stress test exposed as the
    primary failure mode, and it is now the default Deliveroo template. The
    V1 prompt is still available via ``hint == "deliveroo_v1"`` if a caller
    explicitly needs the classic layout (with miles/minutes).

    Args:
        hint: Optional caller-supplied platform hint, e.g. ``"uber_eats"`` or
            ``"deliveroo"``.

    Returns:
        The full prompt string to send as the user message to the vision model.
    """
    hint_clean = (hint or "").strip().lower()[:64]

    if hint_clean == "deliveroo":
        return DELIVEROO_V2_EXTRACTION_PROMPT

    if hint_clean == "deliveroo_v1":
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
    "raw_text must contain ONLY text visible in the image, never JSON syntax. "
    "Emit the JSON now."
)
