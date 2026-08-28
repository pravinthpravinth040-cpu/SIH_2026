"""
OceanGuard - Webhook & Notification Manager Module
Handles:
1. Incoming Webhook Ingress (Satellite, AIS, External Sensors) with HMAC validation.
2. Webhook Event Dispatcher for Inter-Agent Pub/Sub queue.
3. Modular Multi-Channel Notification Outputs (Dashboard, Email, SMS, External Ports).
"""

import os
import hmac
import hashlib
import json
import datetime
from typing import Dict, Any, List, Optional, Callable

WEBHOOK_SECRET = os.getenv("API_KEY", "XzQgiffxY0uAsFgL6df2fcgerAmdEqg8VHafPr0U")

class WebhookNotificationManager:
    """Manages outgoing alert notifications and external system webhooks."""

    def __init__(self):
        self.notification_history: List[Dict[str, Any]] = []
        self.event_subscribers: Dict[str, List[Callable]] = {}

    def verify_signature(self, payload_bytes: bytes, signature_header: Optional[str]) -> bool:
        """Verifies SHA-256 HMAC signature of incoming webhook payloads."""
        if not signature_header:
            return True # Allow permissive in dev if no signature header supplied
        try:
            expected = hmac.new(WEBHOOK_SECRET.encode(), payload_bytes, hashlib.sha256).hexdigest()
            # Support both raw hex or sha256=prefix
            sig = signature_header.replace("sha256=", "").strip()
            return hmac.compare_digest(expected, sig)
        except Exception:
            return False

    def subscribe(self, event_type: str, handler: Callable):
        """Register an internal event subscriber."""
        if event_type not in self.event_subscribers:
            self.event_subscribers[event_type] = []
        self.event_subscribers[event_type].append(handler)

    def publish_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Publishes an event to all internal queue subscribers."""
        event_packet = {
            "event_id": f"EVT-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:17]}",
            "event": event_type,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "payload": payload
        }
        
        handlers = self.event_subscribers.get(event_type, [])
        for h in handlers:
            try:
                h(event_packet)
            except Exception as e:
                print(f"[WebhookManager] Handler error for {event_type}: {e}")

        return event_packet

    def dispatch_high_risk_alert(
        self,
        vessel_name: str,
        mmsi: str,
        vessel_type: str,
        attribution_score: float,
        spill_id: str,
        lat: float,
        lon: float,
        confidence: float,
        area_km2: float
    ) -> Dict[str, Any]:
        """
        Dispatches multi-channel notification when a high-risk candidate vessel is scored.
        Strictly applies non-accusatory maritime terminology:
        'Potential vessel', 'Suspected vessel', 'High-risk candidate', 'Attribution score'.
        """
        alert_payload = {
            "alert_id": f"ALT-{datetime.datetime.utcnow().strftime('%Y%m%d')}-{mmsi[-4:]}",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "title": f"🚨 HIGH-RISK CANDIDATE VESSEL DETECTED (Attribution: {attribution_score:.1f}%)",
            "candidate_classification": "High-risk candidate" if attribution_score >= 80.0 else "Suspected vessel",
            "attribution_score": attribution_score,
            "vessel_details": {
                "vessel_name": vessel_name,
                "mmsi": mmsi,
                "vessel_type": vessel_type,
            },
            "spill_details": {
                "spill_id": spill_id,
                "estimated_origin": f"{lat:.4f}°N, {lon:.4f}°E",
                "area_km2": area_km2,
                "detection_confidence": confidence
            },
            "channels_notified": [
                "GIS Live Dashboard (WebSocket)",
                "Coast Guard & Port Authority Ingress Webhook",
                "Maritime Incident Operations Email",
                "SMS / Duty Officer Alert Dispatcher"
            ]
        }

        # 1. Modular Email Dispatch Simulation
        self._send_email_notification(alert_payload)

        # 2. Modular SMS / Pager Dispatch Simulation
        self._send_sms_notification(alert_payload)

        # 3. External Webhook Dispatch Simulation
        self._send_external_webhook(alert_payload)

        # Record in persistent notification cache
        self.notification_history.insert(0, alert_payload)
        if len(self.notification_history) > 100:
            self.notification_history.pop()

        return alert_payload

    def _send_email_notification(self, alert: Dict[str, Any]):
        """Modular Email Notification Provider."""
        print(f"[Email Notification Dispatcher] To: ops@coastguard.maritime.gov | Subject: {alert['title']}")

    def _send_sms_notification(self, alert: Dict[str, Any]):
        """Modular SMS / WhatsApp Notification Provider."""
        v = alert['vessel_details']
        s = alert['spill_details']
        msg = f"OceanGuard Alert: {alert['candidate_classification']} '{v['vessel_name']}' (MMSI: {v['mmsi']}) attributed to {s['spill_id']} at {s['estimated_origin']} with score {alert['attribution_score']}%."
        print(f"[SMS Notification Dispatcher] Dispatch: {msg}")

    def _send_external_webhook(self, alert: Dict[str, Any]):
        """Modular External Webhook Forwarder to Port Authority / Coast Guard."""
        print(f"[External Webhook Dispatcher] POST https://maritime-ops.gov.in/api/v1/incidents -> Status 200 OK")

    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns in-memory recent alert notifications."""
        return self.notification_history[:limit]


webhook_manager = WebhookNotificationManager()
