"""Emergency outer ASGI denial boundary for all legacy analytics routes.

This is a separately reviewable contingency, not a deployed component. It must
be used only through the explicit fallback_app entrypoint after CEO approval.
Preserves every non-analytics ASGI scope untouched.
"""
import json
from uuid import uuid4


def _protected(path, root):
    if not isinstance(path, str):
        return True  # Malformed routing evidence fails closed.
    candidates = [path]
    if isinstance(root, str) and root and path.startswith(root):
        suffix = path[len(root):]
        if not suffix or suffix.startswith("/"):
            candidates.append(suffix or "/")
    for value in candidates:
        clean = "/" + "/".join(piece for piece in value.split("/") if piece)
        folded = clean.casefold()
        if folded == "/analytics" or folded.startswith("/analytics/"):
            return True
    return False


class ContainmentOnly:
    """Defence-in-depth: unconditional deny, independent of identity registry."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        kind = scope.get("type")
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if not _protected(scope.get("path"), scope.get("root_path")):
            await self.app(scope, receive, send)
            return
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        request_id = str(uuid4())
        body = json.dumps({"error": "analytics_containment_active", "request_id": request_id}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"cache-control", b"no-store"),
            (b"x-content-type-options", b"nosniff"),
            (b"x-request-id", request_id.encode("ascii")),
        ]
        await send({"type": "http.response.start", "status": 503, "headers": headers})
        await send({"type": "http.response.body", "body": b"" if scope.get("method") == "HEAD" else body})
