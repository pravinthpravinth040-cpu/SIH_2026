import os
import unittest
from unittest.mock import Mock, patch

from api_service import ExternalAPIError, ExternalAPIService


class ExternalAPIServiceTests(unittest.TestCase):
    def make_service(self):
        with patch.dict(os.environ, {
            "EXTERNAL_API_URL": "https://api.example.test/data",
            "EXTERNAL_API_KEY": "test-only-key",
            "EXTERNAL_API_AUTH_HEADER": "Authorization",
            "EXTERNAL_API_AUTH_SCHEME": "Bearer",
        }, clear=False):
            return ExternalAPIService()

    @patch("api_service.requests.get")
    def test_authenticated_valid_response(self, get):
        service = self.make_service()
        response = Mock(status_code=200)
        response.json.return_value = {"latitude": 18.5, "longitude": 72.5}
        get.return_value = response

        result = service.fetch_location_data(latitude=18.5, longitude=72.5)

        self.assertEqual(result["status"], "available")
        get.assert_called_once()
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer test-only-key")

    @patch("api_service.requests.get")
    def test_invalid_json_is_rejected(self, get):
        service = self.make_service()
        response = Mock(status_code=200)
        response.json.side_effect = ValueError("not json")
        get.return_value = response

        with self.assertRaises(ExternalAPIError):
            service.fetch_location_data(latitude=18.5, longitude=72.5)

    def test_missing_configuration_is_unavailable(self):
        with patch.dict(os.environ, {"EXTERNAL_API_URL": "", "EXTERNAL_API_KEY": ""}, clear=False):
            service = ExternalAPIService()
        self.assertFalse(service.configured)
        self.assertEqual(
            service.fetch_location_data(latitude=18.5, longitude=72.5)["status"],
            "unavailable",
        )


if __name__ == "__main__":
    unittest.main()