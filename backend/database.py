import os
import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Load .env configuration
def _load_env():
    env_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env()

DATABASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.getenv("DATABASE_NAME", "oil_spilling")
if not DB_NAME.endswith(".db") and not os.getenv("DATABASE_URL"):
    DATABASE_FILE = os.path.join(DATABASE_DIR, f"{DB_NAME}.db")
else:
    DATABASE_FILE = os.path.join(DATABASE_DIR, DB_NAME if DB_NAME.endswith(".db") else f"{DB_NAME}.db")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_FILE}")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {"connect_timeout": 5}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SatelliteDataset(Base):
    __tablename__ = "satellite_datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    dataset_type = Column(String(50), default="Sentinel-1 SAR") # Sentinel-1 SAR, Uploaded, Historical
    source = Column(String(255), default="Zenodo Sentinel-1 SAR Oil Spill Dataset")
    image_id = Column(String(100), unique=True, index=True)
    file_path = Column(String(500), nullable=True)
    preview_url = Column(String(500), nullable=True)
    record_count = Column(Integer, default=1)
    acquisition_date = Column(DateTime, default=datetime.datetime.utcnow)
    lat_min = Column(Float, nullable=True)
    lat_max = Column(Float, nullable=True)
    lon_min = Column(Float, nullable=True)
    lon_max = Column(Float, nullable=True)
    center_lat = Column(Float, default=0.0)
    center_lon = Column(Float, default=0.0)
    geographic_coverage = Column(String(255), default="Global Marine Waters")
    processing_status = Column(String(50), default="Ready") # Ready, Processing, Completed
    is_real_data = Column(Boolean, default=True)
    data_label = Column(String(50), default="REAL DATA") # REAL DATA or SYNTHETIC/DEMO DATA
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    spills = relationship("OilSpill", back_populates="satellite_dataset")

class AISDataset(Base):
    __tablename__ = "ais_datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    dataset_type = Column(String(50), default="MarineCadastre AIS") # MarineCadastre, Uploaded, Synthetic
    source = Column(String(255), default="MarineCadastre.gov AccessAIS")
    file_path = Column(String(500), nullable=True)
    record_count = Column(Integer, default=0)
    date_range_start = Column(DateTime, nullable=True)
    date_range_end = Column(DateTime, nullable=True)
    geographic_coverage = Column(String(255), default="Coastal & Offshore Waters")
    processing_status = Column(String(50), default="Indexed")
    is_real_data = Column(Boolean, default=True)
    data_label = Column(String(50), default="REAL DATA") # REAL DATA or SYNTHETIC/DEMO DATA
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    records = relationship("AISRecord", back_populates="dataset", cascade="all, delete-orphan")

class AISRecord(Base):
    __tablename__ = "ais_records"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("ais_datasets.id"), nullable=True)
    mmsi = Column(String(50), index=True, nullable=False)
    vessel_name = Column(String(255), default="Unknown Vessel")
    vessel_type = Column(String(100), default="Tanker / Cargo")
    timestamp = Column(DateTime, index=True, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    sog = Column(Float, default=0.0) # Speed Over Ground in knots
    cog = Column(Float, default=0.0) # Course Over Ground in degrees
    heading = Column(Float, nullable=True) # Heading in degrees
    imo = Column(String(50), nullable=True)
    callsign = Column(String(50), nullable=True)
    status = Column(String(100), default="Underway Using Engine")
    is_synthetic = Column(Boolean, default=False)
    data_label = Column(String(50), default="REAL DATA")

    dataset = relationship("AISDataset", back_populates="records")

class OilSpill(Base):
    __tablename__ = "oil_spills"

    id = Column(Integer, primary_key=True, index=True)
    spill_id = Column(String(100), unique=True, index=True) # e.g. OS-2026-0825-001
    satellite_dataset_id = Column(Integer, ForeignKey("satellite_datasets.id"), nullable=True)
    detection_date = Column(DateTime, default=datetime.datetime.utcnow)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    region_name = Column(String(255), default="Offshore Marine Zone")
    area_km2 = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0) # 0 to 100%
    severity = Column(String(50), default="HIGH") # LOW, MEDIUM, HIGH, CRITICAL
    spill_type = Column(String(100), default="Surface Crude / Heavy Oil")
    density = Column(String(50), default="Moderate - High")
    spread_km = Column(Float, default=12.5)
    drift_direction = Column(String(50), default="North-East")
    drift_speed_kn = Column(Float, default=1.8)
    model_name = Column(String(100), default="Attention U-Net SAR Segmentor")
    polygon_json = Column(Text, nullable=True) # GeoJSON polygon of detected spill
    bounding_box_json = Column(Text, nullable=True)
    image_url = Column(String(500), nullable=True)
    mask_url = Column(String(500), nullable=True)
    is_real_data = Column(Boolean, default=True)
    data_label = Column(String(50), default="REAL DATA") # REAL DATA or SYNTHETIC/DEMO DATA
    weather_info_json = Column(Text, nullable=True)
    prediction_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    satellite_dataset = relationship("SatelliteDataset", back_populates="spills")
    attributions = relationship("VesselAttribution", back_populates="spill", cascade="all, delete-orphan")

class VesselAttribution(Base):
    __tablename__ = "vessel_attributions"

    id = Column(Integer, primary_key=True, index=True)
    spill_id = Column(String(100), ForeignKey("oil_spills.spill_id"), index=True)
    mmsi = Column(String(50), index=True, nullable=False)
    vessel_name = Column(String(255), default="Unknown Vessel")
    vessel_type = Column(String(100), default="Oil Tanker")
    distance_km = Column(Float, default=0.0)
    time_diff_min = Column(Float, default=0.0)
    course_deg = Column(Float, default=0.0)
    speed_kn = Column(Float, default=0.0)
    heading_deg = Column(Float, default=0.0)
    passed_through_spill = Column(Boolean, default=False)
    trajectory_consistency = Column(Float, default=0.0) # 0 to 100%
    attribution_score = Column(Float, default=0.0) # Overall Probability 0 to 100%
    status = Column(String(50), default="LOW") # HIGH, MEDIUM, LOW, PROBABLE CAUSE
    rank = Column(Integer, default=1)
    trajectory_json = Column(Text, nullable=True) # Array of points [[lat, lon, ts, sog], ...]
    scoring_breakdown_json = Column(Text, nullable=True)
    is_synthetic = Column(Boolean, default=False)
    data_label = Column(String(50), default="REAL DATA")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    spill = relationship("OilSpill", back_populates="attributions")


class ExternalDataRecord(Base):
    __tablename__ = "external_api_data"

    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(String(100), index=True, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    status = Column(String(30), nullable=False, default="unavailable")
    data_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
