"""
Multi-objective route optimizer for assigning cleanup locations to
multiple volunteer teams / vehicles — inspired by the vehicle-routing
literature (Lu, Pu & Han, 2020's bi-objective cost/workload model;
Hussain et al., 2024's CO2-from-distance formula), implemented as a
lightweight, fully offline, fully testable heuristic:

    1. SWEEP CLUSTERING (Gillett & Miller, 1974) — sort every location
       by its compass bearing from the depot, then split that ordering
       into `num_vehicles` contiguous, size-balanced groups. This is a
       classic, simple way to get geographically sensible, workload-
       balanced clusters without needing a heavy optimization library.
    2. NEAREST-NEIGHBOR CONSTRUCTION — build an initial visiting order
       within each cluster.
    3. 2-OPT LOCAL SEARCH — repeatedly uncross route segments to reduce
       total distance. Standard, well-understood TSP improvement step.
    4. CO2 ESTIMATE — distance-based formula in the same style as
       Hussain et al. (2024): fuel-consumption-per-100km × distance ×
       emission-factor-per-liter. Constants are cited, not invented.

HONEST SCOPE: this is a heuristic, not a proven-optimal solver (true
multi-objective VRP is NP-hard). For the handful of locations a real
cleanup campaign would have (tens, not thousands), this gives sensible,
genuinely useful routes in milliseconds — which is what matters for a
volunteer coordinator, not mathematical optimality.
"""

import math
from itertools import combinations


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def _bearing_from_depot(depot, point) -> float:
    """Compass bearing (0-360 deg) from depot to point, for the sweep."""
    lat1, lon1 = math.radians(depot[0]), math.radians(depot[1])
    lat2, lon2 = math.radians(point[0]), math.radians(point[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def sweep_cluster(depot, points: list, num_vehicles: int) -> list:
    """
    Split `points` (list of (lat, lon, label) tuples) into `num_vehicles`
    contiguous, roughly-equal-size clusters by compass bearing from the
    depot. Returns a list of `num_vehicles` point-lists.
    """
    num_vehicles = max(1, min(num_vehicles, len(points))) if points else 1
    if not points:
        return [[] for _ in range(num_vehicles)]

    ordered = sorted(points, key=lambda p: _bearing_from_depot(depot, (p[0], p[1])))
    clusters = [[] for _ in range(num_vehicles)]
    base_size = len(ordered) // num_vehicles
    remainder = len(ordered) % num_vehicles

    idx = 0
    for v in range(num_vehicles):
        size = base_size + (1 if v < remainder else 0)
        clusters[v] = ordered[idx: idx + size]
        idx += size
    return clusters


def _route_distance(depot, route: list) -> float:
    """Total round-trip distance: depot -> each point in order -> depot."""
    if not route:
        return 0.0
    total = haversine_km(depot[0], depot[1], route[0][0], route[0][1])
    for a, b in zip(route, route[1:]):
        total += haversine_km(a[0], a[1], b[0], b[1])
    total += haversine_km(route[-1][0], route[-1][1], depot[0], depot[1])
    return total


def nearest_neighbor_route(depot, points: list) -> list:
    """Greedy nearest-neighbor construction heuristic."""
    remaining = list(points)
    route = []
    current = depot
    while remaining:
        nxt = min(remaining, key=lambda p: haversine_km(current[0], current[1], p[0], p[1]))
        route.append(nxt)
        remaining.remove(nxt)
        current = (nxt[0], nxt[1])
    return route


def two_opt(depot, route: list, max_iterations: int = 200) -> list:
    """Classic 2-opt: repeatedly reverse a segment if it shortens the route."""
    if len(route) < 4:
        return route

    best = list(route)
    best_dist = _route_distance(depot, best)
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i, j in combinations(range(len(best)), 2):
            if j - i < 1:
                continue
            candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
            cand_dist = _route_distance(depot, candidate)
            if cand_dist < best_dist - 1e-9:
                best, best_dist = candidate, cand_dist
                improved = True
    return best


def estimate_co2_kg(distance_km: float, fuel_l_per_100km: float = 25.0,
                     emission_factor_kg_per_l: float = 2.62) -> float:
    """
    CO2 estimate in the style of Hussain et al. (2024): fuel consumption
    per 100km x distance x emission factor per liter of diesel. Defaults
    (25 L/100km for a truck under ~16 tons, 2.62 kg CO2/L diesel) are the
    same published constants cited in that paper, not invented here —
    override them if you have better local figures.
    """
    liters_used = (distance_km / 100.0) * fuel_l_per_100km
    return round(liters_used * emission_factor_kg_per_l, 2)


def optimize_routes(depot: tuple, locations: list, num_vehicles: int,
                     fuel_l_per_100km: float = 25.0,
                     emission_factor_kg_per_l: float = 2.62) -> dict:
    """
    Main entry point.

    Args:
        depot: (lat, lon) — starting/ending point for every vehicle.
        locations: list of (lat, lon, label) tuples to visit.
        num_vehicles: number of volunteer teams / vehicles available.

    Returns a dict with per-vehicle routes, distances, CO2 estimates,
    and workload-balance metrics (max-min distance gap across vehicles —
    the "workload balance" objective from Lu, Pu & Han, 2020).
    """
    clusters = sweep_cluster(depot, locations, num_vehicles)

    vehicle_routes = []
    for i, cluster in enumerate(clusters):
        if not cluster:
            vehicle_routes.append({
                "vehicle": i + 1, "stops": [], "distance_km": 0.0, "co2_kg": 0.0,
            })
            continue
        initial = nearest_neighbor_route(depot, cluster)
        improved = two_opt(depot, initial)
        dist = round(_route_distance(depot, improved), 2)
        vehicle_routes.append({
            "vehicle": i + 1,
            "stops": improved,
            "distance_km": dist,
            "co2_kg": estimate_co2_kg(dist, fuel_l_per_100km, emission_factor_kg_per_l),
        })

    distances = [r["distance_km"] for r in vehicle_routes]
    total_distance = round(sum(distances), 2)
    total_co2 = round(sum(r["co2_kg"] for r in vehicle_routes), 2)
    workload_balance_km = round(max(distances) - min(distances), 2) if distances else 0.0

    return {
        "vehicle_routes": vehicle_routes,
        "total_distance_km": total_distance,
        "total_co2_kg": total_co2,
        "workload_balance_km": workload_balance_km,
        "num_locations": len(locations),
    }
