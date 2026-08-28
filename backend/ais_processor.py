import os
import io
import math
import json
import csv
import datetime
from typing import List, Dict, Any, Optional, Tuple
from dateutil import parser as date_parser

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two points in kilometers."""
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def parse_ais_csv(csv_content: str, filename: str = "uploaded_ais.csv") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parses MarineCadastre CSV or standard maritime AIS CSV data.
    Returns parsed records and dataset summary metadata.
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    records = []
    
    # Standardize column headers
    field_map = {}
    if reader.fieldnames:
        for field in reader.fieldnames:
            normalized = field.strip().lower().replace("_", "").replace(" ", "")
            if normalized in ["mmsi", "vesselid", "shipid", "id"]:
                field_map["mmsi"] = field
            elif normalized in ["lat", "latitude", "y"]:
                field_map["lat"] = field
            elif normalized in ["lon", "long", "longitude", "x"]:
                field_map["lon"] = field
            elif normalized in ["basedatetime", "timestamp", "datetime", "time", "date"]:
                field_map["timestamp"] = field
            elif normalized in ["sog", "speed", "speedoverground"]:
                field_map["sog"] = field
            elif normalized in ["cog", "course", "courseoverground"]:
                field_map["cog"] = field
            elif normalized in ["heading", "head", "trueheading"]:
                field_map["heading"] = field
            elif normalized in ["vesselname", "shipname", "name"]:
                field_map["vessel_name"] = field
            elif normalized in ["vesseltype", "shiptype", "type"]:
                field_map["vessel_type"] = field
            elif normalized in ["imo", "imonumber"]:
                field_map["imo"] = field
            elif normalized in ["callsign"]:
                field_map["callsign"] = field
            elif normalized in ["status", "navstatus", "navigationstatus"]:
                field_map["status"] = field

    min_lat, max_lat = 90.0, -90.0
    min_lon, max_lon = 180.0, -180.0
    earliest_dt, latest_dt = None, None
    unique_mmsis = set()

    for row in reader:
        try:
            mmsi_val = row.get(field_map.get("mmsi", "MMSI"), "").strip()
            if not mmsi_val:
                continue

            lat_val = float(row.get(field_map.get("lat", "LAT"), 0.0))
            lon_val = float(row.get(field_map.get("lon", "LON"), 0.0))
            if lat_val == 0.0 and lon_val == 0.0:
                continue

            raw_time = row.get(field_map.get("timestamp", "BaseDateTime"), "").strip()
            if raw_time:
                try:
                    dt = date_parser.parse(raw_time)
                except Exception:
                    dt = datetime.datetime.utcnow()
            else:
                dt = datetime.datetime.utcnow()

            sog_val = float(row.get(field_map.get("sog", "SOG"), 0.0) or 0.0)
            cog_val = float(row.get(field_map.get("cog", "COG"), 0.0) or 0.0)
            
            raw_hdg = row.get(field_map.get("heading", "Heading"), "")
            heading_val = float(raw_hdg) if raw_hdg and raw_hdg != "511" and raw_hdg != "511.0" else cog_val
            
            vessel_name = row.get(field_map.get("vessel_name", "VesselName"), "").strip() or f"VESSEL-{mmsi_val[-4:]}"
            vessel_type = row.get(field_map.get("vessel_type", "VesselType"), "").strip() or "Commercial Vessel"
            imo = row.get(field_map.get("imo", "IMO"), "").strip()
            callsign = row.get(field_map.get("callsign", "CallSign"), "").strip()
            status = row.get(field_map.get("status", "Status"), "").strip() or "Underway"

            record = {
                "mmsi": mmsi_val,
                "vessel_name": vessel_name,
                "vessel_type": vessel_type,
                "timestamp": dt,
                "lat": lat_val,
                "lon": lon_val,
                "sog": sog_val,
                "cog": cog_val,
                "heading": heading_val,
                "imo": imo,
                "callsign": callsign,
                "status": status,
                "is_synthetic": False,
                "data_label": "REAL DATA"
            }
            records.append(record)

            min_lat = min(min_lat, lat_val)
            max_lat = max(max_lat, lat_val)
            min_lon = min(min_lon, lon_val)
            max_lon = max(max_lon, lon_val)
            unique_mmsis.add(mmsi_val)

            if earliest_dt is None or dt < earliest_dt:
                earliest_dt = dt
            if latest_dt is None or dt > latest_dt:
                latest_dt = dt

        except Exception:
            continue

    summary = {
        "record_count": len(records),
        "unique_vessels": len(unique_mmsis),
        "date_range_start": earliest_dt.isoformat() if earliest_dt else None,
        "date_range_end": latest_dt.isoformat() if latest_dt else None,
        "lat_min": min_lat if min_lat <= max_lat else None,
        "lat_max": max_lat if min_lat <= max_lat else None,
        "lon_min": min_lon if min_lon <= max_lon else None,
        "lon_max": max_lon if min_lon <= max_lon else None,
        "is_real_data": True,
        "data_label": "REAL DATA",
        "source": "MarineCadastre AccessAIS / Ingested CSV"
    }

    return records, summary

def parse_ais_json(json_content: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parses JSON AIS data array or GeoJSON object."""
    data = json.loads(json_content)
    records = []
    
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if "features" in data and isinstance(data["features"], list):
            for feat in data["features"]:
                props = feat.get("properties", {})
                geom = feat.get("geometry", {})
                coords = geom.get("coordinates", [0, 0])
                item = dict(props)
                item["lon"] = coords[0] if len(coords) > 0 else 0.0
                item["lat"] = coords[1] if len(coords) > 1 else 0.0
                items.append(item)
        elif "records" in data:
            items = data["records"]
        else:
            items = [data]

    min_lat, max_lat = 90.0, -90.0
    min_lon, max_lon = 180.0, -180.0
    earliest_dt, latest_dt = None, None
    unique_mmsis = set()

    for item in items:
        try:
            mmsi_val = str(item.get("mmsi") or item.get("MMSI") or item.get("id") or "")
            if not mmsi_val:
                continue

            lat_val = float(item.get("lat") or item.get("latitude") or item.get("LAT") or 0.0)
            lon_val = float(item.get("lon") or item.get("longitude") or item.get("LON") or 0.0)
            if lat_val == 0.0 and lon_val == 0.0:
                continue

            raw_time = str(item.get("timestamp") or item.get("BaseDateTime") or item.get("time") or "")
            if raw_time:
                try:
                    dt = date_parser.parse(raw_time)
                except Exception:
                    dt = datetime.datetime.utcnow()
            else:
                dt = datetime.datetime.utcnow()

            sog_val = float(item.get("sog") or item.get("SOG") or item.get("speed") or 0.0)
            cog_val = float(item.get("cog") or item.get("COG") or item.get("course") or 0.0)
            heading_val = float(item.get("heading") or item.get("Heading") or cog_val)
            vessel_name = str(item.get("vessel_name") or item.get("VesselName") or item.get("name") or f"VSL-{mmsi_val[-4:]}")
            vessel_type = str(item.get("vessel_type") or item.get("VesselType") or item.get("type") or "Commercial Vessel")
            
            is_synthetic = bool(item.get("is_synthetic", False))
            data_label = "SYNTHETIC/DEMO DATA" if is_synthetic else "REAL DATA"

            record = {
                "mmsi": mmsi_val,
                "vessel_name": vessel_name,
                "vessel_type": vessel_type,
                "timestamp": dt,
                "lat": lat_val,
                "lon": lon_val,
                "sog": sog_val,
                "cog": cog_val,
                "heading": heading_val,
                "imo": str(item.get("imo", "")),
                "callsign": str(item.get("callsign", "")),
                "status": str(item.get("status", "Underway")),
                "is_synthetic": is_synthetic,
                "data_label": data_label
            }
            records.append(record)

            min_lat = min(min_lat, lat_val)
            max_lat = max(max_lat, lat_val)
            min_lon = min(min_lon, lon_val)
            max_lon = max(max_lon, lon_val)
            unique_mmsis.add(mmsi_val)

            if earliest_dt is None or dt < earliest_dt:
                earliest_dt = dt
            if latest_dt is None or dt > latest_dt:
                latest_dt = dt

        except Exception:
            continue

    summary = {
        "record_count": len(records),
        "unique_vessels": len(unique_mmsis),
        "date_range_start": earliest_dt.isoformat() if earliest_dt else None,
        "date_range_end": latest_dt.isoformat() if latest_dt else None,
        "lat_min": min_lat if min_lat <= max_lat else None,
        "lat_max": max_lat if min_lat <= max_lat else None,
        "lon_min": min_lon if min_lon <= max_lon else None,
        "lon_max": max_lon if min_lon <= max_lon else None,
        "is_real_data": not any(r["is_synthetic"] for r in records),
        "data_label": "SYNTHETIC/DEMO DATA" if any(r["is_synthetic"] for r in records) else "REAL DATA",
        "source": "JSON Ingestion"
    }

    return records, summary

def filter_vessels_around_spill(
    records: List[Dict[str, Any]],
    spill_lat: float,
    spill_lon: float,
    spill_time: datetime.datetime,
    radius_km: float = 35.0,
    time_window_hours: float = 6.0
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filters AIS records geographically and temporally around an oil spill.
    Returns records grouped by MMSI (vessel trajectories).
    """
    vessels_tracks = {}
    time_delta = datetime.timedelta(hours=time_window_hours)
    min_time = spill_time - time_delta
    max_time = spill_time + time_delta

    for rec in records:
        ts = rec["timestamp"]
        if isinstance(ts, str):
            ts = date_parser.parse(ts)

        # Check temporal window (optional check, or within time window)
        # If timestamp is within window:
        if ts < min_time or ts > max_time:
            continue

        # Check distance to spill
        dist = haversine_distance_km(spill_lat, spill_lon, rec["lat"], rec["lon"])
        if dist <= radius_km:
            mmsi = rec["mmsi"]
            if mmsi not in vessels_tracks:
                vessels_tracks[mmsi] = []
            
            clean_rec = dict(rec)
            clean_rec["distance_to_spill_km"] = round(dist, 2)
            clean_rec["time_diff_minutes"] = round((ts - spill_time).total_seconds() / 60.0, 1)
            vessels_tracks[mmsi].append(clean_rec)

    # Sort each vessel track chronologically
    for mmsi in vessels_tracks:
        vessels_tracks[mmsi].sort(key=lambda r: r["timestamp"] if isinstance(r["timestamp"], datetime.datetime) else date_parser.parse(r["timestamp"]))

    return vessels_tracks
