"""Public-client Sign in with ChatGPT. HALO owns its registration and tokens.

Never reads Codex/Pi credentials. OAuth URLs and token responses are deliberately
not included in diagnostics. Each issued client is a separate account/workspace.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import secrets
import tempfile
import time
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import jwt
from filelock import FileLock

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {PLAN_SCOPE}"


class ChatGPTAuthError(RuntimeError):
    """Safe to show to users; contains no credentials or authorization URL."""


class SessionExpiredError(ChatGPTAuthError):
    pass


def default_auth_path() -> Path:
    if value := os.environ.get("HALO_CHATGPT_AUTH_FILE"):
        return Path(value).expanduser()
    root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return root / "halo" / "chatgpt.json"


class CredentialStore:
    """Atomic JSON replacement; cross-process lock covers rotating-token refresh."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_auth_path()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock", timeout=60, mode=0o600)

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"ext_agent_host_id": f"urn:uuid:{uuid.uuid4()}", "accounts": {}, "active": None}
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict) or not isinstance(data.get("accounts"), dict):
                raise ValueError
            return data
        except (ValueError, OSError) as exc:
            raise ChatGPTAuthError(
                "Invalid HALO credential file; restore it or use a new auth file."
            ) from exc

    def write(self, data: dict[str, Any]) -> None:
        fd, name = tempfile.mkstemp(prefix=".chatgpt-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as out:
                json.dump(data, out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @staticmethod
    def selected(data: dict[str, Any], account: str | None = None) -> dict[str, Any]:
        record = data["accounts"].get(account or data.get("active"))
        if record is None:
            raise ChatGPTAuthError("No selected ChatGPT account. Run halo-chatgpt login first.")
        return record


def _token_request(form: dict[str, str]) -> dict[str, Any]:
    try:
        response = httpx.post(TOKEN_URL, data=form, timeout=30)
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ChatGPTAuthError(
            "ChatGPT token endpoint could not be reached; retry login/refresh."
        ) from exc
    if response.is_error:
        # Do not echo the server's error description, URL, or request.
        code = body.get("error") if isinstance(body, dict) else None
        if code in {
            "invalid_grant",
            "invalid_refresh_token",
            "token_expired",
            "refresh_token_expired",
            "refresh_token_invalidated",
            "refresh_token_reused",
        }:
            raise SessionExpiredError("ChatGPT session/code expired or revoked; sign in again.")
        raise ChatGPTAuthError(f"ChatGPT token exchange rejected (HTTP {response.status_code}).")
    if not isinstance(body, dict):
        raise ChatGPTAuthError("Token endpoint returned an invalid token set.")
    if form["grant_type"] == "authorization_code" and not isinstance(body.get("scope"), str):
        raise ChatGPTAuthError("Token endpoint did not confirm granted scopes.")
    scopes = str(body.get("scope", "")).split()
    if not body.get("access_token"):
        if (
            form["grant_type"] == "refresh_token"
            or PLAN_SCOPE in scopes
            or "offline_access" in scopes
        ):
            raise ChatGPTAuthError("Token endpoint returned incomplete credentials.")
        return body  # Validated identity-only sign-in may be retained, but cannot infer.
    if (
        not isinstance(body["access_token"], str)
        or str(body.get("token_type", "")).lower() != "bearer"
    ):
        raise ChatGPTAuthError("Token endpoint did not return a Bearer token.")
    if (form["grant_type"] == "refresh_token" or "offline_access" in scopes) and not body.get(
        "refresh_token"
    ):
        raise ChatGPTAuthError("Token endpoint did not return a rotating refresh token.")
    return body


def _validated_identity(id_token: str, client_id: str, nonce: str) -> dict[str, Any]:
    try:
        key = jwt.PyJWKClient(JWKS_URL, timeout=15).get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token,
            key.key,
            algorithms=["RS256"],
            issuer=ISSUER,
            audience=client_id,
            options={"require": ["iss", "aud", "exp", "sub", "nonce"]},
        )
        if not isinstance(claims["sub"], str) or not claims["sub"]:
            raise ValueError
        if not secrets.compare_digest(str(claims["nonce"]), nonce):
            raise ValueError
        return claims
    except (jwt.PyJWTError, ValueError) as exc:
        raise ChatGPTAuthError(
            "ChatGPT ID token failed signature/identity/nonce validation."
        ) from exc


def _apply_tokens(record: dict[str, Any], tokens: dict[str, Any]) -> None:
    if tokens.get("access_token"):
        try:
            expires_in = float(tokens["expires_in"])
            if not math.isfinite(expires_in) or expires_in <= 0:
                raise ValueError
        except (KeyError, ValueError, TypeError) as exc:
            raise ChatGPTAuthError("Token endpoint returned an invalid expiry.") from exc
        record.update(access_token=tokens["access_token"], expires_at=time.time() + expires_in)
    for name in ("refresh_token", "id_token"):
        if tokens.get(name):
            record[name] = tokens[name]
    if "scope" in tokens:
        record["scopes"] = str(tokens["scope"]).split()
    if "earliest_refresh_at" in tokens:
        record["earliest_refresh_at"] = tokens["earliest_refresh_at"]
    else:
        record.pop("earliest_refresh_at", None)


def callback_client(params: dict[str, str], *, state: str, client_id: str) -> tuple[str, str]:
    """Validate the transaction before using even an error callback."""
    if not secrets.compare_digest(params.get("state", ""), state):
        raise ChatGPTAuthError("OAuth callback state did not match.")
    if params.get("error"):
        raise ChatGPTAuthError("ChatGPT authorization was denied or could not complete.")
    issued = params.get("client_id", client_id)
    if not issued or issued == "dynamic_agent_client":
        raise ChatGPTAuthError("Registration callback did not supply an issued client ID.")
    if client_id != "dynamic_agent_client" and issued != client_id:
        raise ChatGPTAuthError("Callback changed the selected account's client ID.")
    if not params.get("code"):
        raise ChatGPTAuthError("OAuth callback did not supply a code.")
    return issued, params["code"]


def login(
    store: CredentialStore,
    *,
    new_account: bool = False,
    enable_plan: bool = False,
    timeout: int = 300,
) -> str:
    with store.lock:
        data = store.read()
        store.write(data)  # Persist host identity before opening the browser.
        previous = None if new_account or not data.get("active") else store.selected(data).copy()
        host_id = data["ext_agent_host_id"]
    client_id = (
        previous["client_id"]
        if previous
        else (data.get("pending_client_id") if not new_account else None) or "dynamic_agent_client"
    )
    state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    received: dict[str, str] | None = None

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            nonlocal received
            parsed = urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_error(404)
                return
            values = parse_qs(parsed.query, keep_blank_values=True)
            # Reject ambiguous repeated parameters instead of silently selecting one.
            if any(len(value) != 1 for value in values.values()):
                self.send_error(400)
                return
            candidate = {key: value[0] for key, value in values.items()}
            if not secrets.compare_digest(candidate.get("state", ""), state):
                self.send_error(400)
                return
            received = candidate
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Return to HALO to see the sign-in result. You may close this tab.")

        def log_message(self, format: str, *args: Any) -> None:
            pass  # BaseHTTPServer would otherwise log codes and query parameters.

    with HTTPServer(("127.0.0.1", 0), Callback) as server:
        server.timeout = 1
        redirect = f"http://127.0.0.1:{server.server_port}/auth/callback"
        query = dict(
            client_id=client_id,
            ext_agent_host_id=host_id,
            response_type="code",
            redirect_uri=redirect,
            scope=SCOPES,
            resource=RESOURCE,
            state=state,
            nonce=nonce,
            code_challenge_method="S256",
            code_challenge=challenge,
        )
        if previous:
            if previous.get("id_token"):
                query["id_token_hint"] = previous["id_token"]
        elif client_id == "dynamic_agent_client":
            query["agent_name_hint"] = "HALO"
        if enable_plan:
            query["prompt"] = "consent"
        # Returning URLs contain an ID-token hint: open them, never print them.
        if not webbrowser.open(f"{AUTHORIZE_URL}?{urlencode(query)}"):
            raise ChatGPTAuthError("System browser did not open. Set BROWSER and retry login.")
        deadline = time.monotonic() + timeout
        while received is None and time.monotonic() < deadline:
            server.handle_request()
    if received is None:
        raise ChatGPTAuthError("ChatGPT browser login timed out; run login again.")
    issued, code = callback_client(received, state=state, client_id=client_id)
    if previous is None:
        with store.lock:
            data = store.read()
            data["pending_client_id"] = issued
            store.write(data)
    tokens = _token_request(
        dict(
            grant_type="authorization_code",
            client_id=issued,
            code=code,
            code_verifier=verifier,
            redirect_uri=redirect,
            resource=RESOURCE,
        )
    )
    claims = _validated_identity(str(tokens.get("id_token", "")), issued, nonce)
    if previous and claims["sub"] != previous["subject"]:
        raise ChatGPTAuthError("Sign-in identity changed; use login --new-account instead.")
    record = dict(
        client_id=issued, issuer=ISSUER, subject=claims["sub"], email=claims.get("email"), scopes=[]
    )
    _apply_tokens(record, tokens)
    with store.lock:
        data = store.read()
        existing = data["accounts"].get(issued)
        if existing and existing["subject"] != record["subject"]:
            raise ChatGPTAuthError("Issued client ID belongs to a different saved identity.")
        data["accounts"][issued] = record
        data["active"] = issued
        if data.get("pending_client_id") == issued:
            data.pop("pending_client_id", None)
        store.write(data)
    return issued


def access_token(store: CredentialStore, account: str | None = None) -> str:
    with store.lock:
        data = store.read()
        record = store.selected(data, account)
        if PLAN_SCOPE not in record.get("scopes", []):
            raise ChatGPTAuthError(
                "ChatGPT plan usage is not authorized; run halo-chatgpt login --enable-plan."
            )
        if record.get("access_token") and record.get("expires_at", 0) > time.time() + 30:
            return record["access_token"]
        earliest = record.get("earliest_refresh_at")
        try:
            ready_at = (
                float(earliest)
                if isinstance(earliest, (int, float))
                else datetime.fromisoformat(earliest.replace("Z", "+00:00")).timestamp()
                if isinstance(earliest, str)
                else 0
            )
        except (ValueError, OverflowError):
            ready_at = 0
        if ready_at > time.time():
            if record.get("access_token") and record.get("expires_at", 0) > time.time():
                return record["access_token"]
            raise ChatGPTAuthError("ChatGPT session cannot refresh yet; try again shortly.")
        if not record.get("refresh_token"):
            raise ChatGPTAuthError("ChatGPT session is signed out/expired; run halo-chatgpt login.")
        try:
            tokens = _token_request(
                dict(
                    grant_type="refresh_token",
                    client_id=record["client_id"],
                    refresh_token=record["refresh_token"],
                    resource=RESOURCE,
                )
            )
        except SessionExpiredError:
            for key in ("access_token", "refresh_token", "expires_at", "earliest_refresh_at"):
                record.pop(key, None)
            store.write(data)
            raise
        _apply_tokens(record, tokens)
        store.write(data)
        if PLAN_SCOPE not in record.get("scopes", []):
            raise ChatGPTAuthError("Refreshed grant no longer permits ChatGPT plan usage.")
        return record["access_token"]


def logout(store: CredentialStore) -> bool:
    """Clear local tokens even if remote revocation cannot be confirmed."""
    confirmed = False
    with store.lock:
        data = store.read()
        record = store.selected(data)
        try:
            if record.get("refresh_token"):
                discovery = httpx.get(DISCOVERY_URL, timeout=15)
                discovery.raise_for_status()
                endpoint = discovery.json()["revocation_endpoint"]
                parsed = urlparse(endpoint)
                if parsed.scheme != "https" or parsed.netloc != "auth.openai.com":
                    raise ValueError
                for attempt in range(3):
                    try:
                        response = httpx.post(
                            endpoint,
                            data=dict(
                                token=record["refresh_token"],
                                token_type_hint="refresh_token",
                                client_id=record["client_id"],
                            ),
                            timeout=15,
                        )
                    except httpx.HTTPError:
                        time.sleep(0.5 * (2**attempt))
                        continue
                    if response.status_code == 200:
                        confirmed = True
                        break
                    if response.status_code < 500:
                        break
                    time.sleep(0.5 * (2**attempt))
            else:
                confirmed = True
        except (httpx.HTTPError, KeyError, ValueError):
            pass
        for key in (
            "access_token",
            "refresh_token",
            "id_token",
            "expires_at",
            "scopes",
            "earliest_refresh_at",
        ):
            record.pop(key, None)
        store.write(data)
    return confirmed
