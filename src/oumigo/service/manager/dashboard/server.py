"""The OumiGo console: the manager's web UI (Workers, GPU Utilization).

Its own FastAPI app / uvicorn process, fault-isolated from the control plane and router:
a render hang here must never touch inference or heartbeats. It reads the fleet from the
control plane over loopback (`/workers`, `/metrics/*`); a background task mirrors the
metrics for the GPU chart.

Its only write path is a close request in the manager's state-dir auth store, which the
running manager acts on (as it does for `oumigo manager node close`); it never calls a
control-plane write endpoint.

It binds loopback only and is reached through an SSH tunnel. Sign-in: `MG_ADMIN_USER` +
`MG_ADMIN_PASSWORD_HASH` (see `admin_account`). Only a signed-in admin can close workers;
with either variable missing there is no admin, so the console is view-only and offers
the Create Admin Account page, which writes the account into the manager's `.env`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from oumigo.common.env import read_env_file, set_env_values
from oumigo.common.http import loopback_verify
from oumigo.service.manager.admin_account import (
    ENV_ADMIN_PASSWORD_HASH,
    ENV_ADMIN_USER,
    AdminAccount,
    hash_password,
    login_problems,
    password_problems,
    verify_password,
)
from oumigo.service.manager.console import _fmt_transport
from oumigo.service.manager.credentials import AuthStore
from oumigo.service.manager.dashboard import tokens as tk
from oumigo.service.manager.dashboard import web
from oumigo.service.manager.dashboard.aggregate import gpu_util_series
from oumigo.service.manager.dashboard.rollup import TIERS, RollupStore
from oumigo.service.manager.dashboard.security import (
    COOKIE_NAME,
    CSRF_FIELD,
    CSRF_HEADER,
    LOCAL_HOSTS,
    Session,
    Sessions,
    SignInLimiter,
    host_of,
    is_loopback,
    same_origin,
)
from oumigo.service.manager.dashboard.source import MetricMirror

log = logging.getLogger("oumigo.service.manager.dashboard")

VLLM_START_METRIC = "vllm:start_timestamp"
CLOSABLE_STATES = ("registering", "initializing", "serving")
READ_ONLY_NOTE = ("Close needs a signed-in admin, and there is no admin account yet. "
                  "Create it with Create Admin Account (top right) to enable Close.")


def create_dashboard_app(
    control_url: str,
    *,
    poll_interval_s: float = 5.0,
    state_dir: Path | None = None,
    env_file: Path | None = None,
    bind_host: str = "127.0.0.1",
) -> FastAPI:
    """Build the console app.

    `state_dir` is the manager's auth state (close requests, and the key that signs
    sessions); `env_file` the manager's `.env`, where Create Admin Account writes;
    `bind_host` must be a loopback address (ValueError otherwise): the console is
    reached through an SSH tunnel only.
    """
    if not is_loopback(bind_host):
        raise ValueError(f"the web console binds loopback only, not {bind_host!r}")
    mirror = MetricMirror(control_url, prefixes=("worker:", "gpu:", *tk.KINDS.values()))
    store = AuthStore.at(state_dir) if state_dir is not None else AuthStore()
    rollups = RollupStore.at(state_dir) if state_dir is not None else RollupStore()

    async def pull(client: httpx.AsyncClient) -> None:
        await mirror.refresh(client)
        try:
            rollups.ingest(mirror.snapshot())
        except Exception as exc:  # noqa: BLE001 - the live charts must keep working
            log.warning("GPU rollup update failed: %s", exc)
    sessions = Sessions.from_secret(store.secret)
    account = AdminAccount.from_env()
    limiter = SignInLimiter()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # No keep-alive reuse: the poll interval outlives uvicorn's idle keep-alive, so a
        # pooled connection would be closed server-side by the next pull.
        async with httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=0),
            verify=loopback_verify(control_url),  # built-in TLS: own manager over loopback
        ) as client:
            app.state.client = client
            await pull(client)  # seed once so the first page isn't empty
            stop = asyncio.Event()

            async def poll_loop() -> None:
                while not stop.is_set():
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=poll_interval_s)
                    except asyncio.TimeoutError:
                        await pull(client)

            task = asyncio.create_task(poll_loop())
            try:
                yield
            finally:
                stop.set()
                task.cancel()

    app = FastAPI(title="OumiGo console", lifespan=lifespan, docs_url=None,
                  redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def local_host_only(request: Request, call_next):
        # Only requests addressed to localhost are answered: a website open in the
        # admin's browser can't point its own name at 127.0.0.1 and read the console
        # while the tunnel is up (DNS rebinding).
        if host_of(request.headers.get("host")) not in LOCAL_HOSTS:
            return Response("Bad Request: unknown host", status_code=400)
        return await call_next(request)

    # --- sessions ------------------------------------------------------------------

    def control_allowed() -> bool:
        """Closing workers needs the admin account (and, per request, its sign-in)."""
        return account.enabled

    def session_of(request: Request) -> Session | None:
        """The request's valid session: anonymous (it carries the CSRF token of the
        sign-in and account forms), or signed in as the current admin with the current
        password."""
        session = sessions.decode(request.cookies.get(COOKIE_NAME))
        if session is None or not session.login:
            return session
        if (account.enabled and session.login == account.login
                and session.fingerprint == sessions.fingerprint(account.password_hash)):
            return session
        return None  # another login, a changed password, or sign-in now off

    def signed_in(request: Request) -> Session | None:
        """The session a page or API call runs under; None when sign-in is needed. With
        sign-in off every visitor has an (anonymous) session."""
        session = session_of(request)
        if not account.enabled:
            return session or sessions.new()
        return session if session is not None and session.login else None

    def with_cookie(response: Response, session: Session, touch: bool = True) -> Response:
        if touch:
            session.last = datetime.now(timezone.utc).timestamp()
        response.set_cookie(COOKIE_NAME, sessions.encode(session), httponly=True,
                            samesite="strict", path="/")
        return response

    def ensure_session(request: Request) -> Session:
        """The request's session, or a new anonymous one (it carries the CSRF token)."""
        return session_of(request) or sessions.new()

    def check_form(request: Request, session: Session | None, form: dict[str, str]) -> None:
        if not same_origin(request.headers.get("origin"), request.headers.get("host")) \
                or not sessions.check_csrf(session, form.get(CSRF_FIELD)):
            raise HTTPException(403, "The form expired or came from another site; reload "
                                     "the page and try again.")

    async def read_form(request: Request) -> dict[str, str]:
        body = (await request.body()).decode("utf-8", errors="replace")
        return {k: v[0] for k, v in parse_qs(body, keep_blank_values=True).items()}

    def html(text: str, session: Session, status: int = 200, touch: bool = True) -> Response:
        return with_cookie(HTMLResponse(text, status_code=status), session, touch)

    def view(request: Request, title: str, active: str, body: str, *, head: str = "",
             script: str = "") -> Response:
        session = signed_in(request)
        if session is None:
            return RedirectResponse("/signin", status_code=303)
        return html(web.page(title, body, active=active, login=session.login or None,
                             sign_in=account.enabled, csrf=sessions.csrf_token(session),
                             head=head, script=script), session)

    def api_session(request: Request) -> Session:
        session = signed_in(request)
        if session is None:
            raise HTTPException(401, "Sign in again.")
        return session

    # --- pages ---------------------------------------------------------------------

    @app.get("/")
    async def index() -> Response:
        return RedirectResponse("/workers", status_code=303)

    @app.get("/workers")
    async def workers_page(request: Request) -> Response:
        note = None if control_allowed() else READ_ONLY_NOTE
        return view(request, "Workers", "workers", web.workers_body(note),
                    script=web.WORKERS_JS)

    @app.get("/gpu")
    async def gpu_page(request: Request) -> Response:
        return view(request, "GPU Utilization", "gpu",
                    web.chart_body("GPU Utilization", web.GPU_INTRO, web.GPU_CHARTS),
                    head=web.CHART_HEAD,
                    script=web.chart_script("/api/sheets/gpu_util", web.GPU_CHARTS, "%", True))

    @app.get("/tokens")
    async def tokens_page(request: Request) -> Response:
        return view(request, "Token Generation", "tokens",
                    web.chart_body("Token Generation", web.TOKEN_INTRO, web.TOKEN_RATE_CHARTS),
                    head=web.CHART_HEAD,
                    script=web.chart_script("/api/sheets/tokens?mode=rate",
                                            web.TOKEN_RATE_CHARTS, "MT / min", False))

    @app.get("/tokens/cumulative")
    async def tokens_cumulative_page(request: Request) -> Response:
        return view(request, "Token Generation (Cumulative)", "tokens-cumulative",
                    web.chart_body("Token Generation (Cumulative)", web.TOKEN_INTRO,
                                   web.TOKEN_CUMULATIVE_CHARTS),
                    head=web.CHART_HEAD,
                    script=web.chart_script("/api/sheets/tokens?mode=cumulative",
                                            web.TOKEN_CUMULATIVE_CHARTS, "MT", False))

    # --- sign-in -------------------------------------------------------------------

    def signin_page(session: Session, login: str = "", problems: list[str] | None = None,
                    status: int = 200) -> Response:
        token = sessions.csrf_token(session)
        body = web.signin_body(token, login, problems)
        return html(web.page("Sign in", body, sign_in=True, csrf=token), session, status)

    @app.get("/signin")
    async def signin_form(request: Request) -> Response:
        if not account.enabled or signed_in(request) is not None:
            return RedirectResponse("/workers", status_code=303)
        return signin_page(ensure_session(request))

    @app.post("/signin")
    async def signin(request: Request) -> Response:
        if not account.enabled:
            return RedirectResponse("/workers", status_code=303)
        session = session_of(request)
        form = await read_form(request)
        check_form(request, session, form)
        login = form.get("login", "").strip()
        if limiter.locked():
            log.warning("console sign-in refused: locked after repeated wrong passwords")
            locked = ("Sign-in is locked for a while after repeated wrong passwords. "
                      "Try again later.")
            return signin_page(session, login, [locked], status=429)
        if login == account.login and verify_password(form.get("password", ""),
                                                      account.password_hash):
            limiter.succeeded()
            log.warning("console sign-in: %s", login)
            new = sessions.new(login, sessions.fingerprint(account.password_hash))
            return with_cookie(RedirectResponse("/workers", status_code=303), new)
        limiter.failed()
        log.warning("console sign-in failed (login %r)", login)
        return signin_page(session, login, ["Wrong login name or password."], status=401)

    @app.post("/signout")
    async def signout(request: Request) -> Response:
        session = session_of(request)
        check_form(request, session, await read_form(request))
        if session is not None and session.login:
            log.warning("console sign-out: %s", session.login)
        target = "/signin" if account.enabled else "/workers"
        return with_cookie(RedirectResponse(target, status_code=303), sessions.new())

    # --- Create Admin Account ------------------------------------------------------

    def account_form_blocker() -> str | None:
        """Why the account form can't be used here (the page then shows the manual
        steps), or None."""
        if env_file is None:
            return "The manager was started without an env file, so there is nowhere to write."
        directory = env_file.parent
        if not (os.access(directory, os.W_OK)
                and (not env_file.exists() or os.access(env_file, os.W_OK))):
            return f"The console can't write {env_file} (permissions)."
        in_file = read_env_file(env_file)
        for key in (ENV_ADMIN_USER, ENV_ADMIN_PASSWORD_HASH):
            if key in os.environ and in_file.get(key) != os.environ[key]:
                return (f"{key} is set in the manager's environment (e.g. systemd "
                        f"Environment=), not in {env_file}; the environment would win over "
                        "the file at the next start. Change it where it is set.")
        return None

    def account_page(session: Session, login: str = "", problems: list[str] | None = None,
                     status: int = 200) -> Response:
        blocker = account_form_blocker()
        if blocker is not None:
            body = web.admin_manual_body(blocker, str(env_file) if env_file else None)
        else:
            body = web.admin_form_body(sessions.csrf_token(session), str(env_file), login,
                                       problems)
        return html(web.page("Create Admin Account", body, sign_in=False,
                             csrf=sessions.csrf_token(session)), session, status)

    @app.get("/admin-account")
    async def admin_account_form(request: Request) -> Response:
        if account.enabled:
            raise HTTPException(404)
        return account_page(ensure_session(request), account.login)

    @app.post("/admin-account")
    async def admin_account_create(request: Request) -> Response:
        if account.enabled:
            raise HTTPException(404)
        session = session_of(request)
        form = await read_form(request)
        check_form(request, session, form)
        if account_form_blocker() is not None:
            return account_page(session, status=403)
        login = form.get("login", "").strip()
        password = form.get("password", "")
        problems = login_problems(login) + [f"Password: {p}" for p in
                                            password_problems(password)]
        if password != form.get("password2", ""):
            problems.append("The two passwords differ.")
        if problems:
            return account_page(session, login, problems, status=400)
        assert env_file is not None
        password_hash = hash_password(password)
        set_env_values(env_file, {ENV_ADMIN_USER: login, ENV_ADMIN_PASSWORD_HASH: password_hash})
        os.environ[ENV_ADMIN_USER] = login
        os.environ[ENV_ADMIN_PASSWORD_HASH] = password_hash
        account.login, account.password_hash = login, password_hash
        log.warning("admin account %s created via console; wrote %s", login, env_file)
        return with_cookie(RedirectResponse("/signin", status_code=303), sessions.new())

    # --- data ----------------------------------------------------------------------

    async def control_get(request: Request, path: str) -> dict:
        try:
            resp = await request.app.state.client.get(f"{control_url}{path}", timeout=5.0)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("control plane %s failed: %s", path, exc)
            raise HTTPException(502, "The manager's control plane did not answer.") from None

    @app.get("/api/workers")
    async def api_workers(request: Request) -> dict:
        api_session(request)
        rows = (await control_get(request, "/workers")).get("workers", [])
        latest = (await control_get(request, "/metrics/latest")).get("nodes", [])
        starts = {n["node_id"]: n.get("metrics", {}).get(VLLM_START_METRIC) for n in latest}
        requested = store.close_requests()
        now = datetime.now(timezone.utc)
        allowed = control_allowed()
        out = []
        for r in sorted(rows, key=lambda r: r.get("seq", 0)):
            node_id = r["node_id"]
            closing = (bool(r.get("closing")) or node_id in requested) and \
                r["state"] not in ("stopped", "failed")
            out.append({
                "name": r.get("name") or node_id[:8],
                "node_id": node_id,
                "address": r.get("address") or "-",
                "port": r.get("port") if r.get("port") is not None else "-",
                "gpu": (r.get("capabilities") or {}).get("gpu") or "-",
                "backend": r.get("backend") or "-",
                "transport": _fmt_transport(r),
                "state": r["state"],
                "model": r.get("profile") or r.get("model") or "-",
                "task": r.get("task") or "-",
                "start": _fmt_utc(starts.get(node_id)),
                "last_seen": f"{now.timestamp() - r['last_seen']:.0f}s ago",
                "closing": closing,
                "closable": allowed and not closing and r["state"] in CLOSABLE_STATES,
            })
        return {"workers": out, "control": allowed,
                "control_note": None if allowed else READ_ONLY_NOTE,
                "generated_at": now.strftime("%Y-%m-%d %H:%M:%S")}

    @app.post("/api/workers/{node_id}/close")
    async def api_close(node_id: str, request: Request) -> JSONResponse:
        session = api_session(request)
        if not same_origin(request.headers.get("origin"), request.headers.get("host")) \
                or not sessions.check_csrf(session, request.headers.get(CSRF_HEADER)):
            raise HTTPException(403, "The page expired; reload it and try again.")
        if not control_allowed():
            raise HTTPException(403, READ_ONLY_NOTE)
        rows = (await control_get(request, "/workers")).get("workers", [])
        record = next((r for r in rows if r.get("node_id") == node_id), None)
        if record is None:
            raise HTTPException(404, f"No registered worker {node_id}.")
        if record["state"] not in CLOSABLE_STATES:
            raise HTTPException(409, f"{record.get('name')} is {record['state']}; there is "
                                     "nothing to close.")
        by = session.login or "console"
        store.request_close(node_id, by)
        log.warning("close requested for %s (%s) by %s", record.get("name"), node_id, by)
        return JSONResponse({"ok": True, "node_id": node_id})

    @app.get("/api/sheets/gpu_util")
    async def sheet_gpu_util(request: Request, range: str = "60m") -> dict:
        """`range`: 60m (5 s grid, from the live mirror), or a rollup tier: 48h, 10d, 45d."""
        api_session(request)
        now = datetime.now(timezone.utc)
        if range == "60m":
            return gpu_util_series(mirror.snapshot(), now, node_info=mirror.node_info)
        if range not in TIERS:
            raise HTTPException(400, f"range must be one of: 60m, {', '.join(TIERS)}")
        return rollups.series(range, now, node_info=mirror.node_info)

    @app.get("/api/sheets/tokens")
    async def sheet_tokens(request: Request, range: str = "60m", mode: str = tk.RATE) -> dict:
        """Tokens per worker: `mode` rate (MT / min) or cumulative (MT); `range` 60m (5 s
        grid, from the live mirror) or a rollup tier: 48h, 10d, 45d."""
        api_session(request)
        if mode not in (tk.RATE, tk.CUMULATIVE):
            raise HTTPException(400, "mode must be rate or cumulative")
        now = datetime.now(timezone.utc)
        if range == "60m":
            return tk.live_series(mirror.snapshot(), now, mode, node_info=mirror.node_info)
        if range not in TIERS:
            raise HTTPException(400, f"range must be one of: 60m, {', '.join(TIERS)}")
        return rollups.token_series(range, now, mode, node_info=mirror.node_info)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "control_url": control_url,
            "buffered_points": len(mirror),
            "last_pull_ok": mirror.last_ok.strftime("%Y-%m-%d %H:%M:%S") if mirror.last_ok else None,
            "last_error": mirror.last_error,
            "sign_in": account.enabled,
        }

    return app


def _fmt_utc(epoch: float | None) -> str:
    if not epoch:
        return "-"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def run_dashboard(
    host: str,
    port: int,
    control_url: str,
    *,
    poll_interval_s: float = 10.0,
    verbose: bool = False,
    state_dir: Path | None = None,
    env_file: Path | None = None,
) -> None:
    """Run the console in the foreground (blocks until stopped)."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        app = create_dashboard_app(control_url, poll_interval_s=poll_interval_s,
                                   state_dir=state_dir, env_file=env_file, bind_host=host)
    except ValueError as exc:
        raise SystemExit(f"web console: {exc}") from None
    if not AdminAccount.from_env().enabled:
        log.warning("console has no admin account (%s / %s not set): view-only until one "
                    "is created", ENV_ADMIN_USER, ENV_ADMIN_PASSWORD_HASH)
    log.info("console on %s:%d (control=%s)", host, port, control_url)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
        log_level="info" if verbose else "warning",
    )
