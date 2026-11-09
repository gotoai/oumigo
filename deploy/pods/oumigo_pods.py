#!/usr/bin/env python3
"""Run an OumiGo worker on a rented GPU: find a machine, watch it join, delete it.

Runs on the manager host, as the manager's service user, with the manager's Python
(it uses the `oumigo` CLI of the same install to mint and revoke tokens):

    sudo -u oumigo /opt/oumigo/venv/bin/python oumigo_pods.py up --wait
    sudo -u oumigo /opt/oumigo/venv/bin/python oumigo_pods.py status
    sudo -u oumigo /opt/oumigo/venv/bin/python oumigo_pods.py down

Providers, tried in the order of `search.providers` (or --providers):

    secure.runpod      RunPod Secure Cloud: RunPod's own data centers
    community.runpod   RunPod Community Cloud: vetted third-party hosts
    vast.ai            Vast.ai marketplace: verified hosts, filtered by reliability

`up`     mints a short-lived join token and rents the first machine that fits, provider
         by provider: a listed GPU, CUDA at least the image's, at most
         `search.max_price_per_hour` (or --max-price). If none has one, it searches again
         every `search.retry_interval` seconds for up to `search.timeout` seconds, then
         fails (and revokes the token). Records the pod (never the token)
         in <auth state dir>/pods/<name>.json. --wait blocks until the worker is serving;
         if it fails or times out instead, the pod is deleted. --dry-run only shows what
         each provider would offer.
`status` shows the pod as its provider sees it and the worker as the manager sees it.
`down`   revokes the worker (the router stops sending it requests at once), deletes the
         pod, revokes the join token, and forgets the pod.

Run by hand or from a timer: the commands are the same. Settings come from a YAML file
(see pods.example.yaml); secrets from an env file readable only by the service user:

    RUNPOD_API_KEY=...   RunPod key that may create and delete pods (*.runpod)
    VAST_API_KEY=...     Vast.ai key (vast.ai)
    REGISTRY_TOKEN=...   read-only registry token for a private image (vast.ai; RunPod
                         keeps its own registry credential)
    HF_TOKEN=...         read-only Hugging Face token, passed to the pod

A provider whose secrets are missing is skipped (on vast.ai, REGISTRY_TOKEN too when
`vast.registry_user` is set).
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from oumigo.common.identity import compute_node_id

DEFAULT_CONFIG = Path("/etc/oumigo/pods.yaml")
DEFAULT_SECRETS = Path("/etc/oumigo/pods.env")
PROVIDERS = ("secure.runpod", "community.runpod", "vast.ai")
VAST_GPUS = ["RTX A6000", "A40", "L40", "L40S", "RTX 6000Ada"]


class Fail(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(f"oumigo-pods: {message}")


def log(message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp} {message}", flush=True)


def price(value) -> str:
    return f"${float(value):.3f}/h" if value is not None else "$?/h"


def duration(seconds: float) -> str:
    m, s = divmod(round(seconds), 60)
    return f"{m}m{s:02d}s" if m and s else f"{m}m" if m else f"{s}s"


# --- settings -----------------------------------------------------------------------


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; '#' starts a comment only at the beginning of a line."""
    if not path.exists():
        raise Fail(f"{path} not found (it holds the provider API keys and HF_TOKEN, mode 0600)")
    if path.stat().st_mode & 0o077:
        raise Fail(f"{path} must be readable by its owner only (chmod 600)")
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


class Settings:
    def __init__(self, config_path: Path, secrets_path: Path, providers: list[str] | None = None,
                 max_price: float | None = None, retry_interval: float | None = None,
                 search_timeout: float | None = None) -> None:
        cfg = yaml.safe_load(config_path.read_text()) or {}
        worker, token, search = (cfg.get(k) or {} for k in ("worker", "join_token", "search"))
        self.name: str = cfg.get("name") or "oumigo-worker"
        self.image: str = _required(cfg, "image", config_path)
        self.gpu_count: int = int(cfg.get("gpu_count", 1))
        self.disk_gb: int = int(cfg.get("disk_gb", 80))
        self.min_cuda: str | None = (str(cfg["min_cuda_version"]) if cfg.get("min_cuda_version")
                                     else cuda_from_tag(self.image))
        self.providers: list[str] = list(providers or search.get("providers") or PROVIDERS)
        if unknown := [p for p in self.providers if p not in PROVIDERS]:
            raise Fail(f"unknown provider {', '.join(unknown)}; known: {', '.join(PROVIDERS)}")
        bound = max_price if max_price is not None else search.get("max_price_per_hour")
        self.max_price: float | None = float(bound) if bound else None  # 0 or unset: no bound
        self.retry_interval = float(retry_interval if retry_interval is not None
                                    else search.get("retry_interval", 30))
        self.search_timeout = float(search_timeout if search_timeout is not None
                                    else search.get("timeout", 1800))   # 0: search once
        self.runpod: dict = cfg.get("runpod") or {}
        self.vast: dict = cfg.get("vast") or {}
        self.manager_url: str = _required(worker, "manager_url", config_path).rstrip("/")
        self.profile: str | None = worker.get("profile")
        self.extra_env: dict[str, str] = {k: str(v) for k, v in (worker.get("env") or {}).items()}
        self.token_ttl: str = str(token.get("ttl", "13h"))
        self.token_uses: int = int(token.get("uses", 3))
        self.manager_config = Path(cfg.get("manager_config", "/etc/oumigo/manager.yaml"))

        self.secrets = read_env_file(secrets_path)
        self.hf_token = self.secrets.get("HF_TOKEN")

        manager = yaml.safe_load(self.manager_config.read_text()) or {}
        control = manager.get("control_plane") or {}
        host = control.get("host", "127.0.0.1")
        self.control_url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{control.get('port', 7014)}"
        self.vllm_port = int((manager.get("worker") or {}).get("vllm_port") or 7001)
        state_dir = Path((manager.get("auth") or {}).get("state_dir") or "/var/lib/oumigo")
        self.state_file = state_dir / "pods" / f"{self.name}.json"

    def within_bound(self, cost) -> bool:
        return self.max_price is None or (cost is not None and float(cost) <= self.max_price + 1e-9)


def cuda_from_tag(image: str) -> str | None:
    """The CUDA version an image tag names, e.g. `...-cu129` -> "12.9"; None if none."""
    match = re.search(r"-cu(\d+)(\d)(?:-|$)", image.rsplit(":", 1)[-1])
    return f"{match.group(1)}.{match.group(2)}" if match else None


def _required(block: dict, key: str, path: Path):
    if not block.get(key):
        raise Fail(f"{path}: `{key}` is required")
    return block[key]


# --- the oumigo CLI (tokens, revocation) -------------------------------------------


def oumigo(s: Settings, *args: str) -> subprocess.CompletedProcess:
    exe = Path(sys.executable).with_name("oumigo")
    cmd = [str(exe) if exe.exists() else shutil.which("oumigo") or "oumigo", "manager", *args,
           "-c", str(s.manager_config)]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def mint_join_token(s: Settings, label: str) -> tuple[str, str]:
    done = oumigo(s, "token", "create", "--ttl", s.token_ttl, "--uses", str(s.token_uses),
                  "--label", label)
    match = re.search(r"id:\s*(\S+)", done.stderr)
    if done.returncode or not match or not done.stdout.strip():
        raise Fail(f"could not mint a join token: {done.stderr.strip()}")
    return match.group(1), done.stdout.strip()


# --- providers -----------------------------------------------------------------------
#
# Each provider offers: `key`, `missing` (the secrets `up` needs but lacks: skipped),
# `offers()` for --dry-run, `create(env)` returning {"pod_id", "gpu", "where", "price"}
# or None when nothing fits, `get(pod_id)` returning {"status", "state", "gpu", "price"}
# (state: starting | running | stopped) or None once the pod is gone, and `delete(pod_id)`.


def _request(provider: str, method: str, url: str, key: str | None, **kwargs) -> httpx.Response:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        return httpx.request(method, url, headers=headers, timeout=60, **kwargs)
    except httpx.RequestError as exc:
        raise Fail(f"{provider} API unreachable: {exc}") from None


def _error(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}: {resp.text[:300]}"
    if isinstance(body, dict):
        body = body.get("error") or body.get("msg") or body
    return f"HTTP {resp.status_code}: {body}"


class RunPod:
    REST = "https://rest.runpod.io/v1"
    GRAPHQL = "https://api.runpod.io/graphql"
    DEPLOY = """mutation Deploy($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) { id costPerHr machine { gpuDisplayName location } }
    }"""
    PRICES = "query { gpuTypes { id securePrice communityPrice } }"

    def __init__(self, s: Settings, cloud: str) -> None:
        self.s, self.cloud = s, cloud            # SECURE | COMMUNITY
        self.name = f"{cloud.lower()}.runpod"
        self.key = s.secrets.get("RUNPOD_API_KEY")
        self.gpu_types: list[str] = list(s.runpod.get("gpu_types") or [])
        self.data_centers: list[str] = list(s.runpod.get("data_centers") or [])
        self.registry_auth_id: str | None = s.runpod.get("registry_auth_id")
        self.missing = [] if self.key else ["RUNPOD_API_KEY"]

    def _graphql(self, query: str, variables: dict | None = None) -> tuple[dict, str]:
        resp = _request(self.name, "POST", self.GRAPHQL, self.key,
                        json={"query": query, "variables": variables or {}})
        data = resp.json() if resp.content else {}
        errors = "; ".join(e.get("message", "") for e in data.get("errors") or [])
        if not data.get("data") and not errors:
            errors = f"HTTP {resp.status_code}"
        return data.get("data") or {}, errors

    def offers(self) -> list[tuple[str, float | None]]:
        """The listed GPU types within the price bound, in the listed order, with prices.

        Prices are RunPod's list prices; the pod's actual price is checked again once it
        exists. Whether a machine is free shows only when a pod is asked for.
        """
        data, errors = self._graphql(self.PRICES)
        if errors:
            raise Fail(f"{self.name}: cannot read prices: {errors}")
        field = "securePrice" if self.cloud == "SECURE" else "communityPrice"
        listed = {g["id"]: g.get(field) for g in data.get("gpuTypes") or []}
        offers = []
        for gpu in self.gpu_types:
            if gpu not in listed:
                log(f"{self.name}: unknown GPU type {gpu!r}, skipped")
                continue
            cost = listed[gpu] * self.s.gpu_count if listed[gpu] else None
            if not cost:
                continue                          # not offered in this cloud
            if self.s.within_bound(cost):
                offers.append((gpu, cost))
        return offers

    def create(self, env: dict[str, str]) -> dict | None:
        """Create the pod on the first offered GPU type (and data center) with a free machine.

        Uses RunPod's GraphQL API: unlike the REST API, whose CUDA filter is a list of exact
        versions, it takes a *minimum* CUDA version, so hosts with newer drivers qualify too.
        """
        base = {
            "name": self.s.name,
            "imageName": self.s.image,
            "gpuCount": self.s.gpu_count,
            "cloudType": self.cloud,
            "containerDiskInGb": self.s.disk_gb,
            "volumeInGb": 0,
            "ports": "",               # the worker only dials out
            "startSsh": False,
            "env": [{"key": k, "value": v} for k, v in env.items()],
        }
        if self.s.min_cuda:
            base["minCudaVersion"] = self.s.min_cuda
        if self.registry_auth_id:
            base["containerRegistryAuthId"] = self.registry_auth_id
        offers = self.offers()
        if not offers:
            log(f"{self.name}: no listed GPU type within {price(self.s.max_price)}")
            return None
        misses = []
        for gpu, _ in offers:
            for dc in self.data_centers or [None]:
                body = {**base, "gpuTypeId": gpu, **({"dataCenterId": dc} if dc else {})}
                data, errors = self._graphql(self.DEPLOY, {"input": body})
                pod = data.get("podFindAndDeployOnDemand")
                if pod and pod.get("id"):
                    machine = pod.get("machine") or {}
                    if not self.s.within_bound(pod.get("costPerHr")):
                        log(f"{self.name}: pod {pod['id']} costs {price(pod.get('costPerHr'))}, "
                            f"over {price(self.s.max_price)}; deleting it")
                        self.delete(pod["id"])
                        continue
                    return {"pod_id": pod["id"], "gpu": machine.get("gpuDisplayName") or gpu,
                            "where": machine.get("location"), "price": pod.get("costPerHr")}
                if "available" not in errors.lower():  # not a stock miss: a real error
                    raise Fail(f"{self.name}: pod not created on {gpu}: {errors}")
                misses.append(gpu + (f"@{dc}" if dc else ""))
        log(f"{self.name}: no machine free for {', '.join(misses)}")
        return None

    def get(self, pod_id: str) -> dict | None:
        resp = _request(self.name, "GET", f"{self.REST}/pods/{pod_id}", self.key)
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise Fail(f"{self.name}: cannot read pod {pod_id}: {_error(resp)}")
        pod = resp.json()
        status = pod.get("desiredStatus")
        if status == "TERMINATED":
            return None
        return {"status": status, "state": "running" if status == "RUNNING" else "stopped",
                "gpu": (pod.get("gpu") or {}).get("displayName"), "price": pod.get("costPerHr")}

    def delete(self, pod_id: str) -> None:
        resp = _request(self.name, "DELETE", f"{self.REST}/pods/{pod_id}", self.key)
        if resp.status_code not in (200, 204, 404):
            raise Fail(f"{self.name}: pod {pod_id} not deleted ({_error(resp)})")


class Vast:
    API = "https://console.vast.ai/api/v0"

    def __init__(self, s: Settings) -> None:
        self.s, self.name = s, "vast.ai"
        self.key = s.secrets.get("VAST_API_KEY")
        cfg = s.vast
        self.gpu_names: list[str] = list(cfg.get("gpu_names") or VAST_GPUS)
        self.min_gpu_ram_gb = float(cfg.get("min_gpu_ram_gb", 44))
        self.min_reliability = float(cfg.get("min_reliability", 0.98))
        self.min_download_mbps = float(cfg.get("min_download_mbps", 500))
        self.max_download_usd_per_tb = cfg.get("max_download_usd_per_tb", 10)
        self.registry_user: str | None = cfg.get("registry_user")
        self.registry_token = s.secrets.get("REGISTRY_TOKEN")
        # Without the registry login a private image never pulls, and the rental bills on.
        self.missing = ([] if self.key else ["VAST_API_KEY"]) + \
            (["REGISTRY_TOKEN"] if self.registry_user and not self.registry_token else [])

    def offers(self, limit: int = 10) -> list[dict]:
        """Rentable machines that fit, cheapest first (prices include the disk)."""
        query = {
            "verified": {"eq": True}, "external": {"eq": False},
            "rentable": {"eq": True}, "rented": {"eq": False},
            "num_gpus": {"eq": self.s.gpu_count},
            "gpu_name": {"in": self.gpu_names},
            "gpu_ram": {"gte": self.min_gpu_ram_gb * 1000},   # MB
            "reliability": {"gte": self.min_reliability},
            "inet_down": {"gte": self.min_download_mbps},      # Mb/s: weights and image
            "order": [["dph_total", "asc"]], "type": "ondemand", "limit": limit,
            "allocated_storage": self.s.disk_gb,
        }
        if self.s.min_cuda:
            query["cuda_max_good"] = {"gte": float(self.s.min_cuda)}
        if self.s.max_price:
            query["dph_total"] = {"lte": self.s.max_price}
        if self.max_download_usd_per_tb is not None:
            query["inet_down_cost"] = {"lte": float(self.max_download_usd_per_tb) / 1024}  # $/GB
        resp = _request(self.name, "POST", f"{self.API}/bundles/", self.key, json=query)
        if resp.status_code != 200:
            raise Fail(f"{self.name}: offer search failed: {_error(resp)}")
        # The search filters on the price without the disk; the bound is on the total.
        return [o for o in resp.json().get("offers") or [] if self.s.within_bound(o.get("dph_total"))]

    def create(self, env: dict[str, str]) -> dict | None:
        """Rent the cheapest fitting machine; an offer taken meanwhile moves on to the next."""
        body = {
            "client_id": "me",
            "image": self.s.image,
            "env": env,
            "disk": self.s.disk_gb,
            "label": self.s.name,
            "runtype": "args",         # keep the image's entrypoint; no SSH, no open ports
            "cancel_unavail": True,
        }
        if self.registry_user and self.registry_token:
            registry = self.s.image.split("/", 1)[0]
            body["image_login"] = f"-u {self.registry_user} -p {self.registry_token} {registry}"
        offers = self.offers()
        if not offers:
            log(f"{self.name}: no machine fits (GPU {', '.join(self.gpu_names)}, "
                f"within {price(self.s.max_price)})")
            return None
        for offer in offers[:5]:
            resp = _request(self.name, "PUT", f"{self.API}/asks/{offer['id']}/", self.key, json=body)
            data = resp.json() if resp.content and "json" in resp.headers.get("content-type", "") else {}
            if resp.status_code == 200 and data.get("success") and data.get("new_contract"):
                return {"pod_id": str(data["new_contract"]), "gpu": offer.get("gpu_name"),
                        "where": offer.get("geolocation"), "price": offer.get("dph_total")}
            if resp.status_code in (401, 403):
                raise Fail(f"{self.name}: not allowed to rent: {_error(resp)}")
            log(f"{self.name}: offer {offer['id']} not rented ({_error(resp)})")
        log(f"{self.name}: none of the {min(len(offers), 5)} cheapest offers could be rented")
        return None

    def get(self, pod_id: str) -> dict | None:
        resp = _request(self.name, "GET", f"{self.API}/instances/{pod_id}/", self.key,
                        params={"owner": '"me"'})
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise Fail(f"{self.name}: cannot read instance {pod_id}: {_error(resp)}")
        inst = resp.json().get("instances")
        if not inst:
            return None
        actual, intended = inst.get("actual_status"), inst.get("intended_status")
        if actual == "running":
            state = "running"
        elif actual in ("exited", "stopped") or intended == "stopped":
            state = "stopped"
        else:
            state = "starting"         # created, loading (pulling the image), ...
        return {"status": actual or intended or "?", "state": state, "gpu": inst.get("gpu_name"),
                "price": inst.get("dph_total"),
                "detail": " ".join((inst.get("status_msg") or "").split())[:120]}

    def delete(self, pod_id: str) -> None:
        resp = _request(self.name, "DELETE", f"{self.API}/instances/{pod_id}/", self.key, json={})
        if resp.status_code not in (200, 404):
            raise Fail(f"{self.name}: instance {pod_id} not deleted ({_error(resp)})")


def provider(s: Settings, name: str):
    if name == "vast.ai":
        return Vast(s)
    return RunPod(s, "SECURE" if name == "secure.runpod" else "COMMUNITY")


# --- state ---------------------------------------------------------------------------


def load_state(s: Settings) -> dict | None:
    return json.loads(s.state_file.read_text()) if s.state_file.exists() else None


def save_state(s: Settings, state: dict) -> None:
    s.state_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = s.state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.chmod(0o600)
    tmp.replace(s.state_file)


def manager_worker(s: Settings, node_id: str) -> dict | None:
    try:
        workers = httpx.get(f"{s.control_url}/workers", timeout=10).json()
    except (httpx.HTTPError, ValueError):
        return None
    workers = workers.get("workers", workers) if isinstance(workers, dict) else workers
    return next((w for w in workers if w.get("node_id") == node_id), None)


# --- commands ------------------------------------------------------------------------


def cmd_dry_run(s: Settings) -> None:
    print(f"{s.name}: CUDA >= {s.min_cuda or 'any'}, {s.gpu_count} GPU, "
          f"at most {price(s.max_price) if s.max_price else 'any price'}")
    for name in s.providers:
        p = provider(s, name)
        skipped = f" ({' and '.join(p.missing)} not set: `up` skips it)" if p.missing else ""
        if isinstance(p, Vast):                  # Vast.ai's offer search needs no key
            offers = p.offers()
            print(f"{name}: {len(offers)} machine(s) fit, cheapest first{skipped}")
            for o in offers:
                print(f"  {o['gpu_name']:<14} {price(o['dph_total'])}  {o.get('geolocation') or '?'}, "
                      f"CUDA {o.get('cuda_max_good')}, reliability {o.get('reliability', 0):.3f}, "
                      f"{o.get('inet_down', 0):.0f} Mb/s down at "
                      f"${(o.get('inet_down_cost') or 0) * 1024:.1f}/TB")
        elif p.missing:
            print(f"{name}: skipped, {' and '.join(p.missing)} not set")
        else:
            offers = p.offers()
            print(f"{name}: {len(offers)} GPU type(s) within the bound, tried in this order "
                  "(free machines show only when a pod is asked for)")
            for gpu, cost in offers:
                print(f"  {gpu:<34} {price(cost)}")


def cmd_up(s: Settings, wait: bool, wait_timeout: float) -> None:
    state = load_state(s)
    if state and provider(s, state["provider"]).get(state["pod_id"]):
        raise Fail(f"pod {state['pod_id']} ({s.name}) is already up on {state['provider']}; "
                   "run `down` first")
    usable = []
    for name in s.providers:
        p = provider(s, name)
        if p.missing:
            log(f"{name}: skipped, {' and '.join(p.missing)} not set in the secrets file")
        else:
            usable.append(p)
    if not usable:
        raise Fail("no provider has its secrets in the secrets file")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    token_id, token = mint_join_token(s, label=f"pod {s.name} {stamp}")
    log(f"minted join token {token_id} (ttl {s.token_ttl}, {s.token_uses} uses)")

    # The worker's identity, the same on any provider and across container restarts.
    host_id = f"{s.name}-{secrets.token_hex(6)}"
    node_id = compute_node_id(host_id, s.vllm_port)
    env = {"OUMIGO_MANAGER_URL": s.manager_url, "OUMIGO_JOIN_TOKEN": token,
           "OUMIGO_HOST_ID": host_id, **s.extra_env}
    if s.profile:
        env["OUMIGO_MODEL_PROFILE"] = s.profile
    if s.hf_token:
        env["HF_TOKEN"] = s.hf_token
    pod = chosen = None
    try:
        chosen, pod = find_machine(s, usable, env)
    finally:
        if not pod:
            oumigo(s, "token", "revoke", token_id)
            log(f"revoked join token {token_id}")
    save_state(s, {"name": s.name, "provider": chosen.name, "pod_id": pod["pod_id"],
                   "host_id": host_id, "node_id": node_id, "join_token_id": token_id,
                   "profile": s.profile, "gpu": pod["gpu"], "price": pod["price"],
                   "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    log(f"created pod {pod['pod_id']} ({s.name}) on {chosen.name}: {pod['gpu'] or '?'} in "
        f"{pod['where'] or '?'}, {price(pod['price'])}, expected node {node_id}")
    if wait:
        try:
            wait_serving(s, chosen, pod["pod_id"], node_id, wait_timeout)
        except Fail as exc:
            # A pod whose worker never serves only costs money: take it down.
            log(f"{exc.code}; deleting the pod")
            cmd_down(s, None, None)
            raise


def find_machine(s: Settings, usable: list, env: dict[str, str]) -> tuple:
    """Ask each provider in turn; when none has a machine, wait and ask again.

    Stock comes and goes by the minute, so a search that finds nothing is repeated every
    `search.retry_interval` seconds until `search.timeout` seconds have passed.
    """
    start, attempt = time.monotonic(), 1
    while True:
        for p in usable:
            if pod := p.create(env):
                return p, pod
        waited = time.monotonic() - start
        if waited + s.retry_interval > s.search_timeout:
            raise Fail(f"no machine found on {', '.join(p.name for p in usable)} "
                       f"in {attempt} search(es) over {duration(waited)}")
        log(f"no machine yet (search {attempt}, {duration(waited)} of "
            f"{duration(s.search_timeout)}); searching again in {duration(s.retry_interval)}")
        time.sleep(s.retry_interval)
        attempt += 1


def wait_serving(s: Settings, p, pod_id: str, node_id: str, timeout: float) -> None:
    start, last = time.monotonic(), None
    while time.monotonic() - start < timeout:
        worker = manager_worker(s, node_id)
        if worker:
            state = worker.get("state")
        else:
            pod = p.get(pod_id)
            if pod is None or pod["state"] == "stopped":
                detail = f": {pod['detail']}" if pod and pod.get("detail") else ""
                raise Fail(f"pod {pod_id} is {pod['status'] if pod else 'gone'} "
                           f"before the worker joined{detail}")
            state = f"not registered (pod {pod['status']})"
        if state != last:
            log(f"worker {node_id}: {state} (+{time.monotonic() - start:.0f}s)")
            last = state
        if state == "serving":
            return
        if state in ("failed", "lost", "stopped"):
            raise Fail(f"worker {node_id} is {state}; see the pod's logs on {p.name}")
        time.sleep(15)
    raise Fail(f"worker {node_id} not serving after {timeout:.0f}s")


def cmd_status(s: Settings) -> None:
    state = load_state(s)
    if not state:
        print(f"{s.name}: no pod recorded")
        return
    pod = provider(s, state["provider"]).get(state["pod_id"])
    if pod is None:
        print(f"{s.name}: pod {state['pod_id']} no longer exists on {state['provider']}")
    else:
        print(f"{s.name}: pod {state['pod_id']} on {state['provider']}, {pod['status']}, "
              f"{pod['gpu'] or state.get('gpu') or '?'}, {price(pod['price'] or state.get('price'))}, "
              f"created {state['created_at']}")
    worker = manager_worker(s, state["node_id"])
    if worker:
        print(f"worker {state['node_id']}: {worker.get('state')}, {worker.get('served_name')}, "
              f"{worker.get('transport')}, gpu {(worker.get('capabilities') or {}).get('gpu')}")
    else:
        print(f"worker {state['node_id']}: not registered with the manager")


def cmd_down(s: Settings, pod_id: str | None, provider_name: str | None) -> None:
    state = load_state(s) or {}
    if pod_id and pod_id != state.get("pod_id"):
        state = {}                     # a pod this script did not record
    pod_id = pod_id or state.get("pod_id")
    provider_name = provider_name or state.get("provider")
    if not pod_id or not provider_name:
        raise Fail(f"no pod recorded for {s.name}; pass --pod-id and --provider")
    p = provider(s, provider_name)
    if not p.key:
        raise Fail(f"{provider_name}: its API key is not set in the secrets file")

    # Revoke first: the router stops routing to it and drops its tunnels within seconds,
    # so deleting the pod cannot cut a client's request mid-stream.
    # Deleting the pod is what stops the bill, so a failed revoke only warns.
    node_id = state.get("node_id")
    if node_id:
        done = oumigo(s, "node", "revoke", node_id)
        log(f"revoked node {node_id}" if done.returncode == 0
            else f"warning: node revoke failed: {done.stderr.strip()}")
        time.sleep(5)
    else:
        log("warning: the pod's node is unknown, so it is not revoked first")
    p.delete(pod_id)
    log(f"deleted pod {pod_id} on {provider_name}")
    if token_id := state.get("join_token_id"):
        oumigo(s, "token", "revoke", token_id)  # fails harmlessly once the token is spent
    if node_id:
        # The host id is never reused, so the revocation has done its job.
        oumigo(s, "node", "unrevoke", node_id)
    if state:
        s.state_file.unlink(missing_ok=True)
    log(f"{s.name} is down")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--secrets", type=Path, default=DEFAULT_SECRETS)
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("up", help="rent a machine and start the worker on it")
    up.add_argument("--providers", help=f"comma-separated, tried in order (from: {', '.join(PROVIDERS)})")
    up.add_argument("--max-price", type=float, metavar="USD",
                    help="per hour for the whole pod; 0 for no bound")
    up.add_argument("--retry-interval", type=float, metavar="S",
                    help="seconds between searches when no machine fits (default 30)")
    up.add_argument("--search-timeout", type=float, metavar="S",
                    help="give up after searching this long; 0 searches once (default 1800)")
    up.add_argument("--dry-run", action="store_true", help="show what would be tried; rent nothing")
    up.add_argument("--wait", action="store_true", help="block until the worker serves")
    up.add_argument("--wait-timeout", type=float, default=1800, help="seconds (default 1800)")
    sub.add_parser("status", help="show the pod and its worker")
    down = sub.add_parser("down", help="delete the pod")
    down.add_argument("--pod-id", help="a pod not recorded in the state file")
    down.add_argument("--provider", choices=PROVIDERS, help="the provider of --pod-id")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()] \
        if getattr(args, "providers", None) else None
    s = Settings(args.config, args.secrets, providers, getattr(args, "max_price", None),
                 getattr(args, "retry_interval", None), getattr(args, "search_timeout", None))
    if args.command == "up":
        cmd_dry_run(s) if args.dry_run else cmd_up(s, args.wait, args.wait_timeout)
    elif args.command == "status":
        cmd_status(s)
    else:
        cmd_down(s, args.pod_id, args.provider)


if __name__ == "__main__":
    main()
