from typing import Any, Dict, List, Optional


def create_image_record(*, image_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "stub", "image": image_payload}


def get_image_record(*, image_id: str) -> Optional[Dict[str, Any]]:
    return None


def create_detection_record(*, detection_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "stub", "detection": detection_payload}


def update_detection_record(*, detection_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "stub", "detection_id": detection_id, "updates": updates}


def get_detection_record(*, detection_id: str) -> Optional[Dict[str, Any]]:
    return None


def get_detections(*, limit: int = 50) -> List[Dict[str, Any]]:
    return []


def create_processing_job(*, job_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "stub", "job": job_payload}


def update_processing_job(*, job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "stub", "job_id": job_id, "updates": updates}


def get_vessels(*, limit: int = 50) -> List[Dict[str, Any]]:
    return []


def create_vessel_correlation(*, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "stub", "correlation": payload}
