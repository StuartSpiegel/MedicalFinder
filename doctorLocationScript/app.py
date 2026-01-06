import csv
import io
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, Response

load_dotenv()

app = Flask(__name__)

NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
DETAILS_URL_PREFIX = "https://places.googleapis.com/v1/places/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Base type
BASE_TYPES = ["doctor"]
# Additional types (Strategy 2)
EXTRA_MED_TYPES = ["medical_clinic", "hospital"]


@dataclass
class DoctorOffice:
    place_id: str
    name: str
    address: str
    phone: str
    website: str


def miles_to_meters(miles: float) -> float:
    return miles * 1609.344


def clamp_max_results(n: int) -> int:
    # Places API Nearby Search (New) enforces 1..20 inclusive
    return max(1, min(20, n))


def google_headers(api_key: str, field_mask: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": field_mask,
    }


def safe_text(display_name_obj: Any) -> str:
    # New Places API often returns displayName like {"text": "..."}
    if isinstance(display_name_obj, dict):
        return str(display_name_obj.get("text") or "").strip()
    if display_name_obj is None:
        return ""
    return str(display_name_obj).strip()


def nearby_places(
    api_key: str,
    lat: float,
    lng: float,
    radius_meters: float,
    max_results: int,
    included_types: List[str],
) -> List[Dict[str, Any]]:
    """
    One Nearby Search request (max 20 results) for the given center and radius.
    """
    payload = {
        "includedTypes": included_types,
        "maxResultCount": max_results,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": radius_meters,
            }
        },
    }

    # Keep response small; we’ll fetch phone/website via Place Details.
    field_mask = "places.id,places.displayName,places.formattedAddress"
    resp = requests.post(
        NEARBY_URL,
        headers=google_headers(api_key, field_mask),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("places", [])


def place_details(api_key: str, place_id: str) -> Tuple[str, str]:
    """
    Fetch phone + website if available for a single place.
    """
    field_mask = "nationalPhoneNumber,internationalPhoneNumber,websiteUri"
    url = f"{DETAILS_URL_PREFIX}{place_id}"
    resp = requests.get(url, headers=google_headers(api_key, field_mask), timeout=30)
    resp.raise_for_status()
    d = resp.json()

    phone = (d.get("nationalPhoneNumber") or d.get("internationalPhoneNumber") or "").strip()
    website = (d.get("websiteUri") or "").strip()
    return phone, website


# -----------------------
# Strategy 1: Grid Search
# -----------------------

def meters_per_degree_lat() -> float:
    # Approx meters per degree latitude
    return 111_320.0


def meters_per_degree_lng(lat_deg: float) -> float:
    # Approx meters per degree longitude at a given latitude
    return 111_320.0 * math.cos(math.radians(lat_deg))


def generate_grid_points(
    center_lat: float,
    center_lng: float,
    radius_meters: float,
    step_meters: float,
) -> List[Tuple[float, float]]:
    """
    Generate grid points across a square bounding box around the circle.
    We then keep points that fall within (radius + step/2) to cover edges.
    """
    if step_meters <= 0:
        return [(center_lat, center_lng)]

    lat_step_deg = step_meters / meters_per_degree_lat()
    lng_step_deg = step_meters / max(1.0, meters_per_degree_lng(center_lat))

    # Bounding box extents (degrees) around center for the radius
    lat_radius_deg = radius_meters / meters_per_degree_lat()
    lng_radius_deg = radius_meters / max(1.0, meters_per_degree_lng(center_lat))

    lat_min = center_lat - lat_radius_deg
    lat_max = center_lat + lat_radius_deg
    lng_min = center_lng - lng_radius_deg
    lng_max = center_lng + lng_radius_deg

    points: List[Tuple[float, float]] = []

    # Expand a touch to ensure coverage at edges
    lat = lat_min
    while lat <= lat_max + (lat_step_deg * 0.5):
        lng = lng_min
        while lng <= lng_max + (lng_step_deg * 0.5):
            # Keep points that are reasonably within radius (+ half step)
            if haversine_meters(center_lat, center_lng, lat, lng) <= (radius_meters + step_meters * 0.75):
                points.append((lat, lng))
            lng += lng_step_deg
        lat += lat_step_deg

    # Always include center
    points.append((center_lat, center_lng))

    # Dedup close duplicates by rounding
    uniq = {}
    for (a, b) in points:
        key = (round(a, 5), round(b, 5))
        uniq[key] = (a, b)
    return list(uniq.values())


def haversine_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000.0  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def coverage_to_step_meters(radius_meters: float, coverage: str) -> float:
    """
    Controls how dense the grid is.
    - quick:  1 request (center only)
    - standard: moderate grid
    - extensive: denser grid (more API calls)
    """
    c = (coverage or "standard").lower().strip()
    if c == "quick":
        return 0.0  # center only
    if c == "extensive":
        # Dense overlap: smaller step => more points
        return max(400.0, radius_meters / 6.0)
    # standard
    return max(600.0, radius_meters / 4.0)


# ----------------------------------------
# Combined Strategy 1 + 2: Multi-search
# ----------------------------------------

def fetch_doctors_multi(
    api_key: str,
    lat: float,
    lng: float,
    miles: float,
    max_results_per_request: int,
    coverage: str,
    include_extra_types: bool,
    details_sleep_s: float = 0.05,
    nearby_sleep_s: float = 0.0,
    hard_cap_unique: int = 300,
) -> Tuple[List[DoctorOffice], Dict[str, Any]]:
    """
    Multi-pass search:
      - generate grid points (Strategy 1)
      - search multiple place types (Strategy 2)
      - deduplicate by place_id
      - fetch details (phone + website) for each unique place
    """
    radius_meters = miles_to_meters(miles)
    max_results_per_request = clamp_max_results(max_results_per_request)

    included_types = list(BASE_TYPES)
    if include_extra_types:
        included_types.extend(EXTRA_MED_TYPES)

    # Strategy 1: generate grid points
    step_meters = coverage_to_step_meters(radius_meters, coverage)
    points = [(lat, lng)] if step_meters == 0 else generate_grid_points(lat, lng, radius_meters, step_meters)

    # Strategy 2: multiple types. Places API "includedTypes" can accept a list,
    # but we’ll run one request per type per grid point to get broader coverage.
    # This typically yields more unique results than bundling types into one call.
    types_to_run = list(dict.fromkeys(included_types))  # stable unique

    # Collect raw places and dedupe by place_id
    seen_places: Dict[str, Dict[str, Any]] = {}
    nearby_calls = 0

    for (pt_lat, pt_lng) in points:
        for t in types_to_run:
            try:
                places = nearby_places(
                    api_key=api_key,
                    lat=pt_lat,
                    lng=pt_lng,
                    radius_meters=radius_meters,
                    max_results=max_results_per_request,
                    included_types=[t],  # one type per call for better coverage
                )
                nearby_calls += 1
            except requests.HTTPError:
                # Skip failed cell/type; continue.
                continue

            for p in places:
                pid = (p.get("id") or "").strip()
                if not pid:
                    continue
                # Keep first seen; you can also replace to prefer closest point if you later add distance
                if pid not in seen_places:
                    seen_places[pid] = p

            if len(seen_places) >= hard_cap_unique:
                break

            if nearby_sleep_s:
                time.sleep(nearby_sleep_s)

        if len(seen_places) >= hard_cap_unique:
            break

    # Fetch details for each unique place
    results: List[DoctorOffice] = []
    details_calls = 0

    for pid, p in seen_places.items():
        name = safe_text(p.get("displayName"))
        address = (p.get("formattedAddress") or "").strip()

        phone = ""
        website = ""
        try:
            phone, website = place_details(api_key, pid)
            details_calls += 1
        except requests.HTTPError:
            phone, website = "", ""

        if details_sleep_s:
            time.sleep(details_sleep_s)

        results.append(
            DoctorOffice(
                place_id=pid,
                name=name,
                address=address,
                phone=phone,
                website=website,
            )
        )

    meta = {
        "radius_miles": miles,
        "radius_meters": radius_meters,
        "coverage": coverage,
        "grid_points": len(points),
        "types": types_to_run,
        "nearby_calls": nearby_calls,
        "details_calls": details_calls,
        "unique_places": len(results),
        "max_results_per_request": max_results_per_request,
        "hard_cap_unique": hard_cap_unique,
    }
    return results, meta


def persist_to_text(rows: List[DoctorOffice], lat: float, lng: float, miles: float, meta: Dict[str, Any]) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(DATA_DIR, f"doctors_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Run at: {ts}\n")
        f.write(f"Location: lat={lat}, lng={lng}\n")
        f.write(f"Radius: {miles} miles\n")
        f.write(f"Coverage: {meta.get('coverage')}\n")
        f.write(f"Grid points: {meta.get('grid_points')}\n")
        f.write(f"Types: {', '.join(meta.get('types', []))}\n")
        f.write(f"Nearby calls: {meta.get('nearby_calls')}\n")
        f.write(f"Details calls: {meta.get('details_calls')}\n")
        f.write(f"Unique places: {meta.get('unique_places')}\n\n")

        f.write("PlaceID\tName\tAddress\tPhone\tWebsite\n")
        for r in rows:
            f.write(f"{r.place_id}\t{r.name}\t{r.address}\t{r.phone}\t{r.website}\n")
    return path


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/search")
def api_search():
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        return jsonify({"ok": False, "error": "Server missing GOOGLE_MAPS_API_KEY"}), 500

    data = request.get_json(force=True, silent=True) or {}
    try:
        lat = float(data.get("lat"))
        lng = float(data.get("lng"))
        miles = float(data.get("miles", 5))
        # this is still per-request max (1..20)
        max_results = clamp_max_results(int(data.get("max_results", 20)))
        persist = bool(data.get("persist", True))

        coverage = str(data.get("coverage", "standard")).lower().strip()  # quick|standard|extensive
        include_extra_types = bool(data.get("include_extra_types", True))

        # hard cap for safety/cost (total unique results per run)
        hard_cap_unique = int(data.get("hard_cap_unique", 300))
        hard_cap_unique = max(20, min(1000, hard_cap_unique))

    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid input. Expected lat/lng as numbers."}), 400

    try:
        rows, meta = fetch_doctors_multi(
            api_key=api_key,
            lat=lat,
            lng=lng,
            miles=miles,
            max_results_per_request=max_results,
            coverage=coverage,
            include_extra_types=include_extra_types,
            details_sleep_s=0.05,
            nearby_sleep_s=0.0,
            hard_cap_unique=hard_cap_unique,
        )
    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text
        except Exception:
            pass
        return jsonify({"ok": False, "error": f"Places API request failed: {str(e)}", "details": body}), 502

    persisted_path = ""
    if persist:
        persisted_path = persist_to_text(rows, lat, lng, miles, meta)

    return jsonify({
        "ok": True,
        "meta": {"lat": lat, "lng": lng, **meta},
        "persisted_path": persisted_path,
        "rows": [asdict(r) for r in rows],
    })


@app.post("/api/export.csv")
def api_export_csv():
    data = request.get_json(force=True, silent=True) or {}
    rows_in = data.get("rows", [])
    if not isinstance(rows_in, list):
        return jsonify({"ok": False, "error": "rows must be a list"}), 400

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["place_id", "name", "address", "phone", "website"])
    writer.writeheader()
    for item in rows_in:
        if not isinstance(item, dict):
            continue
        writer.writerow({
            "place_id": item.get("place_id", ""),
            "name": item.get("name", ""),
            "address": item.get("address", ""),
            "phone": item.get("phone", ""),
            "website": item.get("website", ""),
        })

    csv_bytes = output.getvalue().encode("utf-8")
    filename = f"doctors_{time.strftime('%Y%m%d-%H%M%S')}.csv"

    return Response(
        csv_bytes,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "8080"))
    app.run(host=host, port=port, debug=True)
