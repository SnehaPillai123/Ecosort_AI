"""
Disposal guidance & environmental impact layer.

This turns a raw classification ("plastic", 94% confidence) into a full
decision-support output:

    Image -> Class -> Confidence -> Disposal -> Environmental Recommendation

...rather than just a label. This is what makes the project a genuine
"AI solution" rather than a plain image classifier for the competition
judges — see the "real-life application" and "AI concept" slides.

Each category has:
    bin     - which waste stream it belongs to
    tip     - a practical disposal tip
    impact  - why correct disposal matters, in plain language
    fact    - an illustrative, order-of-magnitude environmental stat
              (approximate — rounded for presentation impact, not sourced
              precisely; swap in your own cited figures if presenting these
              as hard numbers to judges)
    points  - "Green Score" awarded for correctly identifying/disposing of
              this category, used by the session analytics dashboard

Edit CATEGORY_GUIDE to match the exact class-folder names you train on
(they must match your dataset's folder names).
"""

CATEGORY_ICONS = {
    "plastic": "♻️",
    "paper": "📄",
    "cardboard": "📦",
    "metal": "🔩",
    "glass": "🍾",
    "organic": "🍂",
    "biological": "🍂",
    "trash": "🗑️",
    "battery": "🔋",
    "clothes": "👕",
    "shoes": "👟",
}
DEFAULT_ICON = "❓"

CATEGORY_GUIDE = {
    "plastic": {
        "bin": "Dry / Recyclable Waste (Blue Bin)",
        "tip": "Rinse out any food residue before disposing. Remove caps if "
               "your local facility sorts them separately.",
        "impact": "Plastic can take hundreds of years to break down in a "
                   "landfill. Recycling it cuts demand for new plastic "
                   "production and keeps it out of waterways and soil.",
        "fact": "Recycling roughly 1 kg of plastic can avoid an estimated "
                "1.5–2 kg of CO\u2082-equivalent emissions versus producing it new.",
        "points": 10,
    },
    "paper": {
        "bin": "Dry / Recyclable Waste (Blue Bin)",
        "tip": "Flatten boxes to save space. Keep paper dry — wet or "
               "food-soiled paper often can't be recycled.",
        "impact": "Recycling paper reduces demand for virgin wood pulp, "
                   "helping lower deforestation pressure and the water/energy "
                   "used in papermaking.",
        "fact": "Recycling paper uses an estimated 60% less energy than "
                "making it from raw wood pulp.",
        "points": 10,
    },
    "cardboard": {
        "bin": "Dry / Recyclable Waste (Blue Bin)",
        "tip": "Break down boxes flat. Remove any tape or plastic packaging first.",
        "impact": "Cardboard is one of the most recyclable materials — "
                   "recycling it saves landfill space and significantly "
                   "cuts the energy needed versus producing new cardboard.",
        "fact": "Cardboard can typically be recycled 5–7 times before its "
                "fibers become too short to reuse.",
        "points": 10,
    },
    "metal": {
        "bin": "Dry / Recyclable Waste (Blue Bin)",
        "tip": "Rinse cans before disposal. Sharp edges should be handled carefully.",
        "impact": "Recycling metal (especially aluminum) uses a fraction of "
                   "the energy required to produce it from raw ore, and "
                   "metal can be recycled repeatedly without losing quality.",
        "fact": "Recycling aluminum can use an estimated 90%+ less energy "
                "than producing it from raw bauxite ore.",
        "points": 10,
    },
    "glass": {
        "bin": "Dry / Recyclable Waste (Blue Bin) or Glass Collection Point",
        "tip": "Handle carefully — broken glass should be wrapped before disposal "
               "to protect waste handlers.",
        "impact": "Glass is 100% recyclable and can be reused indefinitely "
                   "without any loss in quality, unlike many plastics.",
        "fact": "Glass can be recycled endlessly — a recycled glass bottle "
                "can be back on a shelf as a new one in about a month.",
        "points": 10,
    },
    "organic": {
        "bin": "Wet / Organic Waste (Green Bin)",
        "tip": "Great for composting. Avoid mixing with plastic packaging.",
        "impact": "Composting organic waste instead of landfilling it "
                   "reduces methane emissions (a potent greenhouse gas) and "
                   "returns nutrients to the soil.",
        "fact": "Food waste left to rot in a landfill produces methane — a "
                "greenhouse gas estimated at 25\u00d7+ more potent than CO\u2082 "
                "over 100 years. Composting avoids this.",
        "points": 8,
    },
    "biological": {  # alias, in case using the Kaggle 12-class dataset
        "bin": "Wet / Organic Waste (Green Bin)",
        "tip": "Great for composting. Avoid mixing with plastic packaging.",
        "impact": "Composting organic waste instead of landfilling it "
                   "reduces methane emissions (a potent greenhouse gas) and "
                   "returns nutrients to the soil.",
        "fact": "Food waste left to rot in a landfill produces methane — a "
                "greenhouse gas estimated at 25\u00d7+ more potent than CO\u2082 "
                "over 100 years. Composting avoids this.",
        "points": 8,
    },
    "trash": {
        "bin": "General / Non-Recyclable Waste (Black Bin)",
        "tip": "This item can't be recycled through standard streams — "
               "consider if a specialized disposal facility applies.",
        "impact": "Correctly routing non-recyclables to general waste — "
                   "instead of the recycling stream — prevents contaminating "
                   "an entire batch of otherwise recyclable material.",
        "fact": "A single incorrectly-sorted item can contaminate an entire "
                "batch of recyclables, sending all of it to landfill instead.",
        "points": 3,
    },
    "battery": {
        "bin": "Hazardous Waste — DO NOT put in regular bins",
        "tip": "Take to a designated e-waste/battery collection point. "
               "Batteries can leak toxic chemicals in landfills.",
        "impact": "Proper battery disposal prevents heavy metals and toxic "
                   "chemicals from leaching into soil and groundwater.",
        "fact": "A single improperly-disposed battery can leak heavy metals "
                "capable of contaminating a large volume of soil or water "
                "over time.",
        "points": 10,
    },
    "clothes": {
        "bin": "Textile Donation/Recycling — not regular trash",
        "tip": "Donate if wearable, or drop at a textile recycling bin. "
               "Textiles take a long time to decompose in landfills.",
        "impact": "Textiles are among the slowest-decomposing landfill "
                   "materials. Donating or recycling clothing reduces the "
                   "fashion industry's landfill footprint.",
        "fact": "Textiles can take decades to centuries to decompose in a "
                "landfill, depending on the fabric.",
        "points": 7,
    },
    "shoes": {
        "bin": "Textile Donation/Recycling — not regular trash",
        "tip": "Donate if wearable, or drop at a shoe/textile recycling "
               "collection point.",
        "impact": "Like clothing, shoes decompose very slowly. Donating "
                   "wearable pairs extends their useful life before they "
                   "ever reach a landfill.",
        "fact": "Synthetic shoe materials can take an estimated 30–40 years "
                "or more to break down in a landfill.",
        "points": 7,
    },
}

DEFAULT_GUIDE = {
    "bin": "General Waste",
    "tip": "Category not recognized in the guidance table — please check "
           "local disposal rules.",
    "impact": "Impact information isn't available for this category yet.",
    "fact": "",
    "points": 1,
}


def get_guidance(category: str) -> dict:
    """Look up disposal guidance (bin, tip, impact, fact, points) for a
    predicted category (case-insensitive)."""
    return CATEGORY_GUIDE.get(category.lower(), DEFAULT_GUIDE)


def get_icon(category: str) -> str:
    """Look up a display icon/emoji for a predicted category (case-insensitive)."""
    return CATEGORY_ICONS.get(category.lower(), DEFAULT_ICON)


UNKNOWN_THRESHOLD = 0.40  # below this, don't present a guess as an answer at all


def confidence_level(confidence: float) -> dict:
    """
    Classify a confidence score into a human-readable level, so the app can
    flag uncertain predictions instead of presenting every guess as equally
    reliable — a small but genuine "AI that knows what it doesn't know"
    touch that's worth highlighting to judges.

    Below UNKNOWN_THRESHOLD, "unknown" is set True — the caller should
    suppress the disposal-guidance card entirely rather than show a
    likely-wrong category with a scary-looking low number next to it.
    A model forced to output *some* class will do so even for a photo of
    a human face or a wall; a hard floor is the honest way to handle that,
    not a lower confidence badge on a wrong answer.
    """
    if confidence >= 0.80:
        return {"label": "High confidence", "emoji": "✅", "warn": False, "unknown": False}
    elif confidence >= 0.50:
        return {"label": "Moderate confidence", "emoji": "🟡", "warn": False, "unknown": False}
    elif confidence >= UNKNOWN_THRESHOLD:
        return {
            "label": "Low confidence",
            "emoji": "⚠️",
            "warn": True,
            "unknown": False,
            "message": "The model isn't very sure about this one — try "
                       "retaking the photo with better lighting, a plainer "
                       "background, or a closer angle on the item.",
        }
    else:
        return {
            "label": "Unable to identify",
            "emoji": "❓",
            "warn": True,
            "unknown": True,
            "message": "I couldn't confidently identify a waste item in this "
                       "photo. Please take a closer image with the object "
                       "clearly visible against a plain background — this "
                       "might not be waste at all, or the item may be "
                       "outside the categories this model was trained on.",
        }
