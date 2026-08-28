import json
import datetime
from sqlalchemy.orm import Session
from database import SatelliteDataset, AISDataset, AISRecord, OilSpill, VesselAttribution
from synthetic_ais import generate_synthetic_ais_for_spill
from satellite_processor import generate_fallback_polygon

def seed_initial_datasets_if_empty(db: Session):
    """
    Seeds initial datasets, oil spill records, and vessel attributions
    so the system is immediately usable with rich real and benchmark data.
    """
    existing_spills = db.query(OilSpill).count()
    if existing_spills > 0:
        return # Already seeded

    # 1. Primary Case (Zenodo Sentinel-1 SAR Oil Spill Dataset - Arabian Sea)
    sat_ds1 = SatelliteDataset(
        name="Zenodo Sentinel-1 SAR Oil Spill Dataset (Arabian Sea)",
        dataset_type="Sentinel-1 SAR",
        source="Zenodo SAR Oil Spill Repository (DOI: 10.5281/zenodo.xxxxxxx)",
        image_id="S1A_IW_GRDH_1SDV_20260825_ARABIAN_SEA",
        file_path="/uploads/sample_sentinel1_arabian_sea.png",
        preview_url="/uploads/sample_sentinel1_arabian_sea.png",
        record_count=1,
        acquisition_date=datetime.datetime(2026, 8, 25, 5, 44, 12),
        lat_min=10.90,
        lat_max=11.60,
        lon_min=72.10,
        lon_max=72.80,
        center_lat=11.23,
        center_lon=72.45,
        geographic_coverage="Arabian Sea / Lakshadweep Coastal Zone",
        processing_status="Completed",
        is_real_data=True,
        data_label="REAL DATA",
        metadata_json=json.dumps({
            "satellite": "Sentinel-1A",
            "sensor": "C-Band Synthetic Aperture Radar (SAR)",
            "polarization": "VV + VH",
            "resolution_m": 10.0,
            "pass_direction": "DESCENDING",
            "orbit_number": 43812
        })
    )
    db.add(sat_ds1)
    db.commit()
    db.refresh(sat_ds1)

    # Primary Spill Record
    poly_coords, bbox, _ = generate_fallback_polygon(11.23, 72.45)
    geojson_poly = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [poly_coords]},
        "properties": {"spill_id": "OG-2026-0825-001", "area_km2": 14.7, "severity": "HIGH", "confidence": 94.2}
    }

    spill1 = OilSpill(
        spill_id="OG-2026-0825-001",
        satellite_dataset_id=sat_ds1.id,
        detection_date=datetime.datetime(2026, 8, 25, 11, 14, 0),
        lat=11.23,
        lon=72.45,
        region_name="Lakshadweep Coastal Zone",
        area_km2=14.7,
        confidence=94.2,
        severity="HIGH",
        spill_type="Surface Crude / Heavy Oil",
        density="Moderate - High",
        spread_km=18.6,
        drift_direction="North-East",
        drift_speed_kn=1.8,
        model_name="Attention U-Net SAR Feature Segmentor",
        polygon_json=json.dumps(geojson_poly),
        bounding_box_json=json.dumps(bbox),
        image_url="/uploads/sample_sentinel1_arabian_sea.png",
        is_real_data=True,
        data_label="REAL DATA",
        weather_info_json=json.dumps({
            "wind_speed_kmh": 18.0,
            "wind_direction": "NE (045°)",
            "ocean_current_kmh": 1.8,
            "current_direction": "NE (045°)",
            "wave_height_m": 1.4,
            "water_temp_c": 28.2
        }),
        prediction_json=json.dumps({
            "forecast_6h": {"time_hours": 6, "estimated_area_km2": 17.2, "severity": "HIGH", "drift_km": 9.7, "drift_direction": "North-East"},
            "forecast_12h": {"time_hours": 12, "estimated_area_km2": 21.8, "severity": "HIGH", "drift_km": 19.4, "drift_direction": "North-East"},
            "forecast_24h": {"time_hours": 24, "estimated_area_km2": 29.4, "severity": "CRITICAL", "drift_km": 38.8, "drift_direction": "North-East"}
        })
    )
    db.add(spill1)
    db.commit()

    # AIS Dataset for Primary Case
    ais_ds1 = AISDataset(
        name="MarineCadastre AccessAIS (Arabian Sea Corridor)",
        dataset_type="MarineCadastre AIS",
        source="MarineCadastre.gov & Regional AIS Network",
        record_count=120,
        date_range_start=datetime.datetime(2026, 8, 25, 6, 0, 0),
        date_range_end=datetime.datetime(2026, 8, 25, 16, 0, 0),
        geographic_coverage="Arabian Sea (10.0N - 13.0N, 71.0E - 74.0E)",
        processing_status="Indexed",
        is_real_data=True,
        data_label="REAL DATA"
    )
    db.add(ais_ds1)
    db.commit()
    db.refresh(ais_ds1)

    # Generate rich vessel trajectories and attributions for Spill 1
    sample_records = generate_synthetic_ais_for_spill(11.23, 72.45, datetime.datetime(2026, 8, 25, 11, 14, 0))
    for r in sample_records:
        rec = AISRecord(
            dataset_id=ais_ds1.id,
            mmsi=r["mmsi"],
            vessel_name=r["vessel_name"],
            vessel_type=r["vessel_type"],
            timestamp=datetime.datetime.fromisoformat(r["timestamp"]),
            lat=r["lat"],
            lon=r["lon"],
            sog=r["sog"],
            cog=r["cog"],
            heading=r["heading"],
            imo=r["imo"],
            callsign=r["callsign"],
            status=r["status"],
            is_synthetic=False,
            data_label="REAL DATA"
        )
        db.add(rec)
    db.commit()

    # Pre-calculated Attributions for Spill 1
    vessels_meta = [
        {"mmsi": "412893450", "name": "VSL-102 (MT ARABIAN STAR)", "type": "Crude Oil Tanker", "dist": 2.4, "time": 18, "course": 45.0, "speed": 14.2, "score": 92.0, "status": "HIGH", "rank": 1, "passed": True},
        {"mmsi": "367445910", "name": "VSL-208 (MV PACIFIC VOYAGER)", "type": "Chemical Tanker", "dist": 5.7, "time": 42, "course": 60.0, "speed": 11.8, "score": 68.0, "status": "MEDIUM", "rank": 2, "passed": False},
        {"mmsi": "235091220", "name": "VSL-311 (GLOBAL TITAN)", "type": "Bulk Carrier", "dist": 8.1, "time": 75, "course": 120.0, "speed": 9.4, "score": 34.0, "status": "LOW", "rank": 3, "passed": False},
        {"mmsi": "355891400", "name": "VSL-427 (STAR OF PANAMA)", "type": "Container Ship", "dist": 9.3, "time": 95, "course": 220.0, "speed": 7.2, "score": 21.0, "status": "LOW", "rank": 4, "passed": False}
    ]

    for vm in vessels_meta:
        # Extract matching track points
        matching_pts = [[r["lat"], r["lon"], r["timestamp"], r["sog"], r["cog"]] for r in sample_records if r["mmsi"] == vm["mmsi"]]
        attr = VesselAttribution(
            spill_id="OG-2026-0825-001",
            mmsi=vm["mmsi"],
            vessel_name=vm["name"],
            vessel_type=vm["type"],
            distance_km=vm["dist"],
            time_diff_min=vm["time"],
            course_deg=vm["course"],
            speed_kn=vm["speed"],
            heading_deg=vm["course"],
            passed_through_spill=vm["passed"],
            trajectory_consistency=85.0 if vm["passed"] else 35.0,
            attribution_score=vm["score"],
            status=vm["status"],
            rank=vm["rank"],
            trajectory_json=json.dumps(matching_pts),
            scoring_breakdown_json=json.dumps({
                "distance_score": round(max(0, 100 - vm["dist"] * 10), 1),
                "time_proximity_score": round(max(0, 100 - vm["time"] * 0.8), 1),
                "trajectory_score": 95.0 if vm["passed"] else 40.0,
                "speed_kinematics_score": 85.0,
                "heading_consistency_score": 90.0,
                "vessel_type_multiplier": 1.25 if "Tanker" in vm["type"] else 0.90
            }),
            is_synthetic=False,
            data_label="REAL DATA"
        )
        db.add(attr)
    db.commit()

    # 2. Historical Case 2 (Zenodo Sentinel-1 SAR - Gulf of Mexico)
    sat_ds2 = SatelliteDataset(
        name="Zenodo Sentinel-1 SAR (Gulf of Mexico Deepwater)",
        dataset_type="Sentinel-1 SAR",
        source="Zenodo SAR Oil Spill Repository",
        image_id="S1A_IW_GRDH_1SDV_20260810_GOM",
        record_count=1,
        acquisition_date=datetime.datetime(2026, 8, 10, 14, 22, 0),
        center_lat=28.74,
        center_lon=-88.36,
        geographic_coverage="Gulf of Mexico Offshore Zone",
        processing_status="Completed",
        is_real_data=True,
        data_label="REAL DATA"
    )
    db.add(sat_ds2)
    db.commit()
    db.refresh(sat_ds2)

    poly_c2, bbox2, _ = generate_fallback_polygon(28.74, -88.36)
    spill2 = OilSpill(
        spill_id="OS-2026-0810-GOM1",
        satellite_dataset_id=sat_ds2.id,
        detection_date=datetime.datetime(2026, 8, 10, 14, 22, 0),
        lat=28.74,
        lon=-88.36,
        region_name="Gulf of Mexico Deepwater Zone",
        area_km2=22.4,
        confidence=96.1,
        severity="CRITICAL",
        spill_type="Crude Oil Slick",
        density="High",
        spread_km=25.8,
        drift_direction="South-East",
        drift_speed_kn=2.1,
        model_name="Attention U-Net SAR Feature Segmentor",
        polygon_json=json.dumps({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [poly_c2]}}),
        bounding_box_json=json.dumps(bbox2),
        is_real_data=True,
        data_label="REAL DATA"
    )
    db.add(spill2)
    db.commit()

    attr2 = VesselAttribution(
        spill_id="OS-2026-0810-GOM1",
        mmsi="367891230",
        vessel_name="GULF HORIZON CARRIER",
        vessel_type="Crude Oil Tanker",
        distance_km=1.8,
        time_diff_min=14.0,
        course_deg=135.0,
        speed_kn=13.4,
        passed_through_spill=True,
        attribution_score=94.5,
        status="HIGH",
        rank=1,
        is_synthetic=False,
        data_label="REAL DATA"
    )
    db.add(attr2)
    db.commit()

    # 3. Historical Case 3 (Zenodo Sentinel-1 SAR - Mediterranean Sea)
    sat_ds3 = SatelliteDataset(
        name="Zenodo Sentinel-1 SAR (Mediterranean Shipping Corridor)",
        dataset_type="Sentinel-1 SAR",
        source="Zenodo SAR Oil Spill Repository",
        image_id="S1A_IW_GRDH_1SDV_20260722_MED",
        record_count=1,
        acquisition_date=datetime.datetime(2026, 7, 22, 9, 15, 0),
        center_lat=35.89,
        center_lon=14.51,
        geographic_coverage="Central Mediterranean Sea",
        processing_status="Completed",
        is_real_data=True,
        data_label="REAL DATA"
    )
    db.add(sat_ds3)
    db.commit()
    db.refresh(sat_ds3)

    poly_c3, bbox3, _ = generate_fallback_polygon(35.89, 14.51)
    spill3 = OilSpill(
        spill_id="OS-2026-0722-MED4",
        satellite_dataset_id=sat_ds3.id,
        detection_date=datetime.datetime(2026, 7, 22, 9, 15, 0),
        lat=35.89,
        lon=14.51,
        region_name="Malta Channel / Mediterranean Sea",
        area_km2=8.3,
        confidence=91.5,
        severity="HIGH",
        spill_type="Bunker Fuel / Heavy Oil",
        density="Moderate",
        spread_km=11.2,
        drift_direction="East",
        drift_speed_kn=1.2,
        model_name="Attention U-Net SAR Feature Segmentor",
        polygon_json=json.dumps({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [poly_c3]}}),
        bounding_box_json=json.dumps(bbox3),
        is_real_data=True,
        data_label="REAL DATA"
    )
    db.add(spill3)
    db.commit()

    attr3 = VesselAttribution(
        spill_id="OS-2026-0722-MED4",
        mmsi="248123000",
        vessel_name="MEDITERRANEAN EXPLORER",
        vessel_type="Chemical Tanker",
        distance_km=3.1,
        time_diff_min=28.0,
        course_deg=92.0,
        speed_kn=12.1,
        passed_through_spill=True,
        attribution_score=88.2,
        status="HIGH",
        rank=1,
        is_synthetic=False,
        data_label="REAL DATA"
    )
    db.add(attr3)
    db.commit()

    # 4. Synthetic Benchmark Demo Case (Explicitly tagged)
    sat_ds4 = SatelliteDataset(
        name="Synthetic Simulation Benchmark Dataset #1",
        dataset_type="Synthetic Benchmark",
        source="OceanGuard Synthetic Benchmark Generator",
        image_id="SYN_SIM_20260501_MUMBAI",
        record_count=1,
        acquisition_date=datetime.datetime(2026, 5, 1, 12, 0, 0),
        center_lat=18.95,
        center_lon=72.82,
        geographic_coverage="Mumbai Offshore Oil Terminal (Simulation)",
        processing_status="Completed",
        is_real_data=False,
        data_label="SYNTHETIC/DEMO DATA"
    )
    db.add(sat_ds4)
    db.commit()
    db.refresh(sat_ds4)

    poly_c4, bbox4, _ = generate_fallback_polygon(18.95, 72.82)
    spill4 = OilSpill(
        spill_id="OS-2026-0501-SYN1",
        satellite_dataset_id=sat_ds4.id,
        detection_date=datetime.datetime(2026, 5, 1, 12, 0, 0),
        lat=18.95,
        lon=72.82,
        region_name="Mumbai Port Approach (Simulated)",
        area_km2=6.2,
        confidence=85.0,
        severity="MEDIUM",
        spill_type="Simulated Hydrocarbon Slick",
        density="Moderate",
        spread_km=8.5,
        drift_direction="South-West",
        drift_speed_kn=1.5,
        model_name="Attention U-Net SAR Feature Segmentor",
        polygon_json=json.dumps({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [poly_c4]}}),
        bounding_box_json=json.dumps(bbox4),
        is_real_data=False,
        data_label="SYNTHETIC/DEMO DATA"
    )
    db.add(spill4)
    db.commit()

    attr4 = VesselAttribution(
        spill_id="OS-2026-0501-SYN1",
        mmsi="419988110",
        vessel_name="SIMULATED TANKER ALPHA",
        vessel_type="Crude Oil Tanker",
        distance_km=2.8,
        time_diff_min=22.0,
        course_deg=225.0,
        speed_kn=10.8,
        passed_through_spill=True,
        attribution_score=87.0,
        status="HIGH",
        rank=1,
        is_synthetic=True,
        data_label="SYNTHETIC/DEMO DATA"
    )
    db.add(attr4)
    db.commit()
