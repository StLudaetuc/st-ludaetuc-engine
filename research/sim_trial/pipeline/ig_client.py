"""Minimal IG REST client for the DEMO environment only.

Login uses IG REST v3 (OAuth token in the response body); this is the method verified from Make on 2026-10-30.
Any base URL other than config.IG_DEMO_BASE is refused, so this client cannot reach a live account."""
import time
import requests

from config import IG_DEMO_BASE


class IGError(RuntimeError):
    pass


class IGDemoClient:
    def __init__(self, api_key, username, password, expected_account_id, base=IG_DEMO_BASE, session=None):
        if base != IG_DEMO_BASE or "demo-api.ig.com" not in base:
            raise IGError(f"refusing non-demo IG base URL: {base}")
        self.base, self.api_key = base, api_key
        self._username, self._password = username, password
        self.expected_account_id = expected_account_id
        self.s = session or requests.Session()
        self.token = None
        self.account_id = None

    # ---------- transport ----------
    def _headers(self, version, auth=True):
        h = {"X-IG-API-KEY": self.api_key, "Version": str(version),
             "Accept": "application/json; charset=UTF-8", "Content-Type": "application/json; charset=UTF-8"}
        if auth:
            h["Authorization"] = f"Bearer {self.token}"
            h["IG-ACCOUNT-ID"] = self.account_id
        return h

    def _req(self, method, path, version, json=None, auth=True, params=None):
        r = self.s.request(method, self.base + path, headers=self._headers(version, auth), json=json,
                           params=params, timeout=30)
        try:
            body = r.json() if r.content else {}
        except ValueError:
            body = {"raw": r.text[:500]}
        if r.status_code >= 400:
            raise IGError(f"{method} {path} -> {r.status_code} {body.get('errorCode', body)}")
        return body

    # ---------- session ----------
    def login(self):
        body = self._req("POST", "/session", 3, json={"identifier": self._username, "password": self._password}, auth=False)
        self.token = body["oauthToken"]["access_token"]
        self.account_id = body["accountId"]
        if self.account_id != self.expected_account_id:
            raise IGError(f"logged-in account {self.account_id} != expected demo account {self.expected_account_id}")
        return self.account_id

    # ---------- reads ----------
    def account(self):
        accts = self._req("GET", "/accounts", 1)["accounts"]
        a = [x for x in accts if x["accountId"] == self.account_id]
        if not a:
            raise IGError("demo account not found in /accounts")
        return a[0]

    def positions(self):
        return self._req("GET", "/positions", 2)["positions"]

    def search(self, term):
        return self._req("GET", "/markets", 1, params={"searchTerm": term})["markets"]

    def market(self, epic):
        return self._req("GET", f"/markets/{epic}", 3)

    # ---------- orders ----------
    def deal(self, epic, expiry, direction, size, level, currency):
        """Marketable limit order, fill-or-kill, netting against any open position (forceOpen=False)."""
        body = {"epic": epic, "expiry": expiry, "direction": direction, "size": round(float(size), 2),
                "orderType": "LIMIT", "level": level, "timeInForce": "FILL_OR_KILL",
                "currencyCode": currency, "forceOpen": False, "guaranteedStop": False}
        ref = self._req("POST", "/positions/otc", 2, json=body)["dealReference"]
        for _ in range(10):
            try:
                return self._req("GET", f"/confirms/{ref}", 1)
            except IGError as e:
                if "404" not in str(e):
                    raise
                time.sleep(1)
        raise IGError(f"no confirmation for dealReference {ref}")
