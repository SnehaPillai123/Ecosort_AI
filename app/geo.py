"""
Zero-API-key location integrations.

    - geocode(): best-effort text -> (lat, lon) via OpenStreetMap's free
      Nominatim service. No key needed, but it's a shared public service —
      be polite (one request at a time, a real User-Agent, cached results
      in db.py's geocode_cache table) and treat failures as "skip the
      pin", never as a blocking error.
    - forecast(): a few days of weather for a lat/lon via Open-Meteo,
      also free and keyless. Used to give a light "good day for a
      cleanup" signal when organizing an event.

Both fail soft: any network hiccup returns None rather than raising, so
a flaky connection during a live demo degrades a nice-to-have, not a
core flow. Neither is called from inside a tight loop — geocoding is
cached, and weather is only fetched when a form is actively being filled.
"""

import requests

import db

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "EcoSortAI-StudentProject/1.0 (AI competition demo; contact via project team)"

WEATHER_CODE_EMOJI = {
    0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
    45: "🌫️", 48: "🌫️",
    51: "🌦️", 53: "🌦️", 55: "🌦️",
    61: "🌧️", 63: "🌧️", 65: "🌧️",
    71: "🌨️", 73: "🌨️", 75: "🌨️",
    80: "🌦️", 81: "🌧️", 82: "⛈️",
    95: "⛈️", 96: "⛈️", 99: "⛈️",
}


def geocode(location_text: str):
    """Best-effort text -> (lat, lon), or None. Checks the shared DB cache
    first so repeated lookups of the same text never hit the network twice."""
    if not location_text or not location_text.strip():
        return None
    location_text = location_text.strip()

    cached = db.get_cached_geocode(location_text)
    if cached:
        return cached

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": location_text, "format": "json", "limit": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        lat, lon = float(results[0]["lat"]), float(results[0]["lon"])
        db.cache_geocode(location_text, lat, lon)
        return (lat, lon)
    except Exception:
        return None


def forecast(lat: float, lon: float, days: int = 3):
    """Returns a list of {date, weather_code, max_temp_c, rain_chance}
    dicts for the next `days` days, or None on failure."""
    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat, "longitude": lon,
                "daily": "weathercode,temperature_2m_max,precipitation_probability_max",
                "forecast_days": days,
                "timezone": "auto",
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()["daily"]
        return [
            {
                "date": data["time"][i],
                "weather_code": data["weathercode"][i],
                "max_temp_c": data["temperature_2m_max"][i],
                "rain_chance": data["precipitation_probability_max"][i],
            }
            for i in range(len(data["time"]))
        ]
    except Exception:
        return None


def weather_emoji(code: int) -> str:
    return WEATHER_CODE_EMOJI.get(code, "🌡️")


def cleanup_suitability(rain_chance) -> str:
    """A light, honestly-hedged read on whether a day looks good for an
    outdoor cleanup — never phrased as a guarantee."""
    if rain_chance is None:
        return ""
    if rain_chance < 20:
        return "✅ Looks like good conditions for an outdoor cleanup"
    if rain_chance < 50:
        return "⚠️ Some rain possible — worth having a backup plan"
    return "🌧️ High rain chance — consider a different date if possible"
