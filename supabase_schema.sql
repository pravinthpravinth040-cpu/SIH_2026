CREATE TABLE IF NOT EXISTS public.satellite_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name TEXT,
    storage_path TEXT,
    image_url TEXT,
    source TEXT,
    satellite TEXT,
    sensor TEXT,
    acquisition_time TIMESTAMPTZ,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID,
    classification TEXT,
    classification_confidence DOUBLE PRECISION,
    oil_spill_detected BOOLEAN,
    segmentation_completed BOOLEAN,
    spill_area DOUBLE PRECISION,
    spill_percentage DOUBLE PRECISION,
    bounding_box JSONB,
    mask_storage_path TEXT,
    mask_url TEXT,
    overlay_storage_path TEXT,
    overlay_url TEXT,
    processing_status TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_detections_image FOREIGN KEY (image_id) REFERENCES public.satellite_images(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS public.processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID,
    status TEXT,
    stage TEXT,
    progress INTEGER,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_jobs_image FOREIGN KEY (image_id) REFERENCES public.satellite_images(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS public.vessels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mmsi TEXT,
    imo TEXT,
    vessel_name TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    speed DOUBLE PRECISION,
    course DOUBLE PRECISION,
    heading DOUBLE PRECISION,
    timestamp TIMESTAMPTZ,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.vessel_correlations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id UUID,
    vessel_id UUID,
    distance_km DOUBLE PRECISION,
    time_difference_hours DOUBLE PRECISION,
    trajectory_score DOUBLE PRECISION,
    risk_score DOUBLE PRECISION,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_vessel_corr_detection FOREIGN KEY (detection_id) REFERENCES public.detections(id) ON DELETE CASCADE,
    CONSTRAINT fk_vessel_corr_vessel FOREIGN KEY (vessel_id) REFERENCES public.vessels(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_satellite_images_created_at ON public.satellite_images(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_detections_created_at ON public.detections(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_image_id ON public.processing_jobs(image_id);
CREATE INDEX IF NOT EXISTS idx_vessels_mmsi ON public.vessels(mmsi);
CREATE INDEX IF NOT EXISTS idx_vessel_correlations_detection_id ON public.vessel_correlations(detection_id);

-- Storage buckets are created in Supabase via the dashboard or SQL when supported.
CREATE TABLE IF NOT EXISTS public.storage_buckets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS policies are created only when the project has the necessary permissions.
-- The default public access pattern is to keep service-role access on the backend only.
ALTER TABLE public.satellite_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.detections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.processing_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vessels ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vessel_correlations ENABLE ROW LEVEL SECURITY;

-- Example RLS policy structure; replace with project-specific rules as needed.
-- CREATE POLICY IF NOT EXISTS "Allow service role to read/write satellite images" ON public.satellite_images
-- FOR ALL USING (true) WITH CHECK (true);
