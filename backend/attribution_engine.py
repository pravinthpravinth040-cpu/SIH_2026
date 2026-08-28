import math
import json
import datetime
from typing import List, Dict, Any, Tuple
from shapely.geometry import Point, Polygon, LineString
from dateutil import parser as date_parser
from ais_processor import haversine_distance_km

VESSEL_TYPE_RISK_WEIGHTS = {
    "crude oil tanker": 1.25,
    "oil tanker": 1.25,
    "chemical tanker": 1.20,
    "products tanker": 1.20,
    "oil / products tanker": 1.20,
    "lpg tanker": 1.10,
    "lng tanker": 1.10,
    "bulk carrier": 0.95,
    "container ship": 0.90,
    "cargo": 0.90,
    "commercial fishing": 0.60,
    "fishing": 0.60,
    "law enforcement / patrol": 0.40,
    "passenger": 0.40,
    "tug": 0.65,
    "other": 0.85
}

def calculate_vessel_attribution(
    spill_id: str,
    spill_lat: float,
    spill_lon: float,
    spill_time: datetime.datetime,
    spill_polygon_geojson: Optional[str],
    vessel_tracks: Dict[str, List[Dict[str, Any]]],
    drift_direction_deg: float = 45.0, # Direction spill is drifting toward
    spill_area_km2: float = 14.7
) -> List[Dict[str, Any]]:
    """
    Computes rigorous multi-factor vessel attribution scores for all nearby vessels.
    Ranks vessels from highest probability to lowest probability.
    """
    # Parse spill polygon if available
    spill_poly = None
    if spill_polygon_geojson:
        try:
            poly_data = json.loads(spill_polygon_geojson)
            if poly_data.get("type") == "Feature":
                coords = poly_data["geometry"]["coordinates"][0]
            elif poly_data.get("type") == "Polygon":
                coords = poly_data["coordinates"][0]
            else:
                coords = []
            if len(coords) >= 3:
                spill_poly = Polygon(coords)
        except Exception:
            spill_poly = None

    attributed_vessels = []

    for mmsi, track in vessel_tracks.items():
        if not track:
            continue

        vessel_name = track[0].get("vessel_name", f"VSL-{mmsi[-4:]}")
        vessel_type = track[0].get("vessel_type", "Commercial Vessel")
        is_synthetic = track[0].get("is_synthetic", False)
        data_label = track[0].get("data_label", "REAL DATA")

        # 1. Find Closest Point of Approach (CPA) to the spill centroid
        min_dist_km = 9999.0
        closest_point = None
        closest_time_diff_min = 0.0

        line_coords = []
        for pt in track:
            pt_lat = pt["lat"]
            pt_lon = pt["lon"]
            line_coords.append((pt_lon, pt_lat)) # Shapely uses (x, y) = (lon, lat)

            dist = haversine_distance_km(spill_lat, spill_lon, pt_lat, pt_lon)
            if dist < min_dist_km:
                min_dist_km = dist
                closest_point = pt
                
                # Time diff in minutes
                ts = pt["timestamp"]
                if isinstance(ts, str):
                    ts = date_parser.parse(ts)
                closest_time_diff_min = abs((ts - spill_time).total_seconds() / 60.0)

        if closest_point is None:
            continue

        # 2. Check Trajectory Intersection with Spill Polygon / Drift Zone
        passed_through = False
        trajectory_line = None
        if len(line_coords) >= 2:
            try:
                trajectory_line = LineString(line_coords)
                if spill_poly and trajectory_line.intersects(spill_poly):
                    passed_through = True
            except Exception:
                pass

        if min_dist_km < 1.2:
            passed_through = True

        # 3. Factor Scores Calculation (0.0 to 100.0 each)
        
        # A. Distance Factor (Exponential decay: 0km -> 100, 3km -> 75, 10km -> 35, 25km -> 5)
        dist_score = max(0.0, min(100.0, 100.0 * math.exp(-0.12 * min_dist_km)))
        
        # B. Temporal Proximity Factor (Gaussian decay around spill detection time)
        # Optimal discharge is 10 to 60 mins before detection
        time_score = max(0.0, min(100.0, 100.0 * math.exp(-0.0003 * (closest_time_diff_min ** 1.8))))

        # C. Trajectory & Intersection Score
        if passed_through:
            traj_score = 95.0 + min(5.0, (1.0 / (min_dist_km + 0.1)))
        elif min_dist_km < 3.5:
            traj_score = 75.0 - (min_dist_km * 8.0)
        elif min_dist_km < 8.0:
            traj_score = 45.0 - (min_dist_km * 3.0)
        else:
            traj_score = max(5.0, 30.0 - min_dist_km)

        # D. Kinematics / Speed Score (Tankers/Vessels discharging often maintain 10-15 kn or have speed variations)
        sog = closest_point.get("sog", 12.0)
        if 8.0 <= sog <= 16.0:
            speed_score = 85.0
        elif 4.0 <= sog < 8.0 or 16.0 < sog <= 20.0:
            speed_score = 65.0
        else:
            speed_score = 40.0

        # E. Heading / Course Consistency Score
        cog = closest_point.get("cog", closest_point.get("heading", 0.0))
        # Check alignment with drift / slick orientation
        angle_diff = abs((cog - drift_direction_deg + 180) % 360 - 180)
        heading_score = max(30.0, 100.0 - (angle_diff * 0.45))

        # F. Vessel Type Risk Multiplier
        norm_type = vessel_type.lower().strip()
        type_multiplier = 0.85
        for k, weight in VESSEL_TYPE_RISK_WEIGHTS.items():
            if k in norm_type:
                type_multiplier = weight
                break

        # 4. Weighted Combined Attribution Probability
        # Weights: Distance (30%), Trajectory Intersection (25%), Temporal (20%), Kinematics (15%), Heading (10%)
        raw_weighted_score = (
            dist_score * 0.30 +
            traj_score * 0.25 +
            time_score * 0.20 +
            speed_score * 0.15 +
            heading_score * 0.10
        )
        
        final_score = min(98.0, max(5.0, raw_weighted_score * type_multiplier))
        attribution_score = round(final_score, 1)

        # Status Classification
        if attribution_score >= 80.0:
            status = "HIGH"
        elif attribution_score >= 50.0:
            status = "MEDIUM"
        else:
            status = "LOW"

        # Trajectory coordinates formatted for mapping
        formatted_trajectory = []
        for pt in track:
            formatted_trajectory.append([
                pt["lat"],
                pt["lon"],
                pt["timestamp"] if isinstance(pt["timestamp"], str) else pt["timestamp"].isoformat(),
                pt.get("sog", 0.0),
                pt.get("cog", 0.0)
            ])

        scoring_breakdown = {
            "distance_score": round(dist_score, 1),
            "time_proximity_score": round(time_score, 1),
            "trajectory_score": round(traj_score, 1),
            "speed_kinematics_score": round(speed_score, 1),
            "heading_consistency_score": round(heading_score, 1),
            "vessel_type_multiplier": round(type_multiplier, 2)
        }

        attributed_vessels.append({
            "spill_id": spill_id,
            "mmsi": mmsi,
            "vessel_name": vessel_name,
            "vessel_type": vessel_type,
            "distance_km": round(min_dist_km, 1),
            "time_diff_min": round(closest_time_diff_min, 1),
            "course_deg": round(closest_point.get("cog", 0.0), 1),
            "speed_kn": round(closest_point.get("sog", 0.0), 1),
            "heading_deg": round(closest_point.get("heading", closest_point.get("cog", 0.0)), 1),
            "passed_through_spill": passed_through,
            "trajectory_consistency": round(traj_score, 1),
            "attribution_score": attribution_score,
            "status": status,
            "trajectory": formatted_trajectory,
            "scoring_breakdown": scoring_breakdown,
            "is_synthetic": is_synthetic,
            "data_label": data_label
        })

    # Sort descending by attribution score
    attributed_vessels.sort(key=lambda x: x["attribution_score"], reverse=True)
    
    # Assign ranks
    for idx, v in enumerate(attributed_vessels):
        v["rank"] = idx + 1

    return attributed_vessels
