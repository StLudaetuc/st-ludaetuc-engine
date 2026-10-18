"""Synthetic, standalone unit controls; no production connection or credentials."""
import asyncio
import json
import unittest
from security.analytics_release.containment_fallback import ContainmentOnly


class FallbackTests(unittest.TestCase):
    def invoke(self, path, root="", method="GET", kind="http"):
        calls, output = [], []

        async def downstream(scope, receive, send):
            calls.append(scope["path"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"untouched"})

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(part):
            output.append(part)

        asyncio.run(ContainmentOnly(downstream)(
            {"type": kind, "path": path, "root_path": root, "method": method}, receive, send
        ))
        return calls, output

    def test_analytics_always_blocked(self):
        for root in ("", "/api", "/x"):
            for endpoint in ("/analytics", "/analytics/export",
                             "/analytics/strategy-report", "/analytics/prediction-accuracy",
                             "/ANALYTICS/export", "//analytics//export"):
                for path in (endpoint, root + endpoint if root else endpoint):
                    with self.subTest(root=root, path=path):
                        calls, output = self.invoke(path, root)
                        self.assertFalse(calls)
                        self.assertEqual(output[0]["status"], 503)
                        self.assertEqual(json.loads(output[-1]["body"])["error"],
                                         "analytics_containment_active")

    def test_unrelated_routes_passthrough(self):
        for root in ("", "/api"):
            for route in ("/health", "/evaluate", "/market-data",
                          "/trade-opened", "/trade-outcome"):
                path = root + route
                calls, output = self.invoke(path, root)
                self.assertEqual(calls, [path])
                self.assertEqual(output[-1]["body"], b"untouched")

    def test_head_and_websocket(self):
        calls, output = self.invoke("/api/analytics/export", "/api", "HEAD")
        self.assertFalse(calls)
        self.assertEqual(output[0]["status"], 503)
        self.assertEqual(output[-1]["body"], b"")
        calls, output = self.invoke("/x/analytics/export", "/x", kind="websocket")
        self.assertFalse(calls)
        self.assertEqual(output, [{"type": "websocket.close", "code": 1008}])

    def test_partial_prefix_not_stripped(self):
        calls, output = self.invoke("/apix/health", "/api")
        self.assertEqual(calls, ["/apix/health"])


if __name__ == "__main__":
    unittest.main()
