import math
import random
import datetime
from typing import List, Dict, Any

VESSEL_TEMPLATES = [
    {
        "name": "MT ARABIAN STAR",
        "type": "Crude Oil Tanker",
        "mmsi": "412893450",
        "imo": "9823145",
        "callsign": "9V8213",
        "base_speed": 14.5,
        "is_culprit_candidate": True
    },
    {
        "name": "MV PACIFIC VOYAGER",
        "type": "Chemical Tanker",
        "mmsi": "367445910",
        "imo": "9451238",
        "callsign": "WDE4821",
        "base_speed": 12.2,
        "is_culprit_candidate": False
    },
    {
        "name": "GLOBAL TITAN",
        "type": "Bulk Carrier",
        "mmsi": "235091220",
        "imo": "9312890",
        "callsign": "MRAX9",
        "base_speed": 11.0,
        "is_culprit_candidate": False
    },
    {
        "name": "STAR OF PANAMA",
        "type": "Container Ship",
        "mmsi": "355891400",
        "imo": "9781204",
        "callsign": "HO3821",
        "base_speed": 18.2,
        "is_culprit_candidate": False
    },
    {
        "name": "OCEAN LEADER",
        "type": "Oil / Products Tanker",
        "mmsi": "477123900",
        "imo": "9621450",
        "callsign": "VRKM4",
        "base_speed": 13.8,
        "is_culprit_candidate": False
    },
    {
        "name": "NORDIC SEA",
        "type": "LPG Tanker",
        "mmsi": "257891230",
        "imo": "9514782",
        "callsign": "LAYG8",
        "base_speed": 15.0,
        "is_culprit_candidate": False
    },
    {
        "name": "COASTAL DEFENDER",
        "type": "Law Enforcement / Patrol",
        "mmsi": "419001230",
        "imo": "8912301",
        "callsign": "VTX11",
        "base_speed": 22.0,
        "is_culprit_candidate": False
    },
    {
        "name": "BLUE MARLIN VII",
        "type": "Commercial Fishing",
        "mmsi": "419992340",
        "imo": "7821903",
        "callsign": "IND92",
        "base_speed": 6.8,
        "is_culprit_candidate": False
    }
]

def generate_synthetic_ais_for_spill(
    center_lat: float,
    center_lon: float,
    detection_time: datetime.datetime,
    num_vessels: int = 8,
    radius_km: float = 25.0
) -> List[Dict[str, Any]]:
    """
    Generates realistic synthetic AIS trajectory data around a spill coordinate.
    - One vessel (culprit candidate) passes directly through the spill ~15-30 mins prior to detection.
    - Other vessels pass at various distances (3km - 20km) on different headings.
    - Explicitly tags each record as DEMO / SYNTHETIC DATA.
    """
    all_records = []
    
    # KM to degrees rough conversion
    lat_deg_per_km = 1.0 / 111.0
    lon_deg_per_km = 1.0 / (111.0 * max(0.2, math.cos(math.radians(center_lat))))
    
    time_window_hours = 4.0 # Generate tracks spanning -2.5h to +1.5h
    interval_minutes = 15
    num_points = int((time_window_hours * 60) / interval_minutes)

    for i in range(min(num_vessels, len(VESSEL_TEMPLATES))):
        vessel = VESSEL_TEMPLATES[i]
        is_culprit = vessel.get("is_culprit_candidate", False) or (i == 0)
        
        if is_culprit:
            # Trajectory passes right through (center_lat, center_lon) at around detection_time - 20 mins
            heading_deg = 48.0 # Heading NE
            course_rad = math.radians(heading_deg)
            speed_kn = vessel["base_speed"]
            speed_kmh = speed_kn * 1.852
            
            # Point in trajectory where it intersects spill
            culprit_index = int(num_points * 0.55)
            
            track_points = []
            for step in range(num_points):
                minutes_from_start = step * interval_minutes
                point_time = detection_time - datetime.timedelta(hours=2.5) + datetime.timedelta(minutes=minutes_from_start)
                
                # Distance along trajectory from intersect point
                delta_minutes = (step - culprit_index) * interval_minutes
                dist_km = (speed_kmh / 60.0) * delta_minutes
                
                # Slight realistic sensor noise
                noise_lat = random.uniform(-0.0008, 0.0008)
                noise_lon = random.uniform(-0.0008, 0.0008)
                
                pt_lat = center_lat + (dist_km * math.cos(course_rad) * lat_deg_per_km) + noise_lat
                pt_lon = center_lon + (dist_km * math.sin(course_rad) * lon_deg_per_km) + noise_lon
                
                sog = max(0.5, speed_kn + random.uniform(-0.4, 0.4))
                cog = (heading_deg + random.uniform(-1.5, 1.5)) % 360
                
                track_points.append({
                    "mmsi": vessel["mmsi"],
                    "vessel_name": vessel["name"],
                    "vessel_type": vessel["type"],
                    "timestamp": point_time.isoformat(),
                    "lat": round(pt_lat, 6),
                    "lon": round(pt_lon, 6),
                    "sog": round(sog, 1),
                    "cog": round(cog, 1),
                    "heading": round(cog, 1),
                    "imo": vessel["imo"],
                    "callsign": vessel["callsign"],
                    "status": "Underway Using Engine",
                    "is_synthetic": True,
                    "data_label": "SYNTHETIC/DEMO DATA"
                })
            all_records.extend(track_points)
        else:
            # Other vessels on parallel or crossing routes at offset distances
            offset_dist_km = 3.5 + (i * 3.2) + random.uniform(-1.0, 1.5)
            heading_deg = (45.0 + (i * 37.0)) % 360
            course_rad = math.radians(heading_deg)
            speed_kn = vessel["base_speed"] + random.uniform(-1.0, 1.0)
            speed_kmh = speed_kn * 1.852
            
            # Offset angle perpendicular or random
            offset_angle_rad = math.radians((heading_deg + 90.0 + (i * 20)) % 360)
            closest_lat = center_lat + (offset_dist_km * math.cos(offset_angle_rad) * lat_deg_per_km)
            closest_lon = center_lon + (offset_dist_km * math.sin(offset_angle_rad) * lon_deg_per_km)
            
            closest_index = random.randint(2, num_points - 3)
            
            track_points = []
            for step in range(num_points):
                minutes_from_start = step * interval_minutes
                point_time = detection_time - datetime.timedelta(hours=2.5) + datetime.timedelta(minutes=minutes_from_start)
                
                delta_minutes = (step - closest_index) * interval_minutes
                dist_km = (speed_kmh / 60.0) * delta_minutes
                
                noise_lat = random.uniform(-0.001, 0.001)
                noise_lon = random.uniform(-0.001, 0.001)
                
                pt_lat = closest_lat + (dist_km * math.cos(course_rad) * lat_deg_per_km) + noise_lat
                pt_lon = closest_lon + (dist_km * math.sin(course_rad) * lon_deg_per_km) + noise_lon
                
                sog = max(0.5, speed_kn + random.uniform(-0.5, 0.5))
                cog = (heading_deg + random.uniform(-2.0, 2.0)) % 360
                
                track_points.append({
                    "mmsi": vessel["mmsi"],
                    "vessel_name": vessel["name"],
                    "vessel_type": vessel["type"],
                    "timestamp": point_time.isoformat(),
                    "lat": round(pt_lat, 6),
                    "lon": round(pt_lon, 6),
                    "sog": round(sog, 1),
                    "cog": round(cog, 1),
                    "heading": round(cog, 1),
                    "imo": vessel["imo"],
                    "callsign": vessel["callsign"],
                    "status": "Underway Using Engine",
                    "is_synthetic": True,
                    "data_label": "SYNTHETIC/DEMO DATA"
                })
            all_records.extend(track_points)
            
    return all_records
