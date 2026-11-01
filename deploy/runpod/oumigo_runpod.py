#!/usr/bin/env python3
"""Run an OumiGo worker as a RunPod pod: create it, watch it join, delete it.

Runs on the manager host, as the manager's service user, with the manager's Python
(it uses the `oumigo` CLI of the same install to mint and revoke tokens):

    sudo -u oumigo /opt/oumigo/venv/bin/python oumigo_runpod.py up --wait
    sudo -u oumigo /opt/oumigo/venv/bin/python oumigo_runpod.py status
    sudo -u oumigo /opt/oumigo/venv/bin/python oumigo_runpod.py down

`up`     mints a short-lived join token, creates the pod with it on the first listed GPU
         type with a free machine, and records the pod (never the token) in
         <auth state dir>/runpod/<name>.json. --wait blocks until the worker is serving;
         if it fails or times out instead, the pod is deleted.
`status` shows the pod as RunPod sees it and the worker as the manager sees it.
`down`   revokes the worker (the router stops sending it requests at once), deletes the
         pod, revokes the join token, and forgets the pod.

Run by hand or from a timer: the commands are the same. Settings come from a YAML file
(see runpod.example.yaml); secrets from an env file readable only by the service user:

    RUNPOD_API_KEY=...       # RunPod API key that may create and delete pods
    HF_TOKEN=...             # read-only Hugging Face token, passed to the pod
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from oumigo.common.identity import compute_node_id

RUNPOD_API = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"
DEFAULT_CONFIG = Path("/etc/oumigo/runpod.yaml")
DEFAULT_SECRETS = Path("/etc/oumigo/runpod.env")


class Fail(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(f"oumigo-runpod: {message}")


def log(message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp} {message}", flush=True)


# --- settings -----------------------------------------------------------------------


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; '#' starts a comment only at the beginning of a line."""
    if not path.exists():
        raise Fail(f"{path} not found (it holds RUNPOD_API_KEY and HF_TOKEN, mode 0600)")
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
    def __init__(self, config_path: Path, secrets_path: Path) -> None:
        cfg = yaml.safe_load(config_path.read_text()) or {}
        pod, worker = cfg.get("pod") or {}, cfg.get("worker") or {}
        token = cfg.get("join_token") or {}
        self.name: str = pod.get("name") or "oumigo-worker"
        self.image: str = _required(pod, "image", config_path)
        self.gpu_types: list[str] = list(_required(pod, "gpu_types", config_path))
        self.gpu_count: int = int(pod.get("gpu_count", 1))
        self.cloud_type: str = pod.get("cloud_type", "SECURE")
        self.min_cuda: str | None = (str(pod["min_cuda_version"]) if pod.get("min_cuda_version")
                                     else cuda_from_tag(self.image))
        self.container_disk_gb: int = int(pod.get("container_disk_gb", 80))
        self.data_centers: list[str] = list(pod.get("data_centers") or [])
        self.registry_auth_id: str | None = pod.get("registry_auth_id")
        self.manager_url: str = _required(worker, "manager_url", config_path).rstrip("/")
        self.profile: str | None = worker.get("profile")
        self.extra_env: dict[str, str] = {k: str(v) for k, v in (worker.get("env") or {}).items()}
        self.token_ttl: str = str(token.get("ttl", "13h"))
        self.token_uses: int = int(token.get("uses", 3))
        self.manager_config = Path(cfg.get("manager_config", "/etc/oumigo/manager.yaml"))

        secrets = read_env_file(secrets_path)
        self.api_key = secrets.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise Fail(f"RUNPOD_API_KEY is not set in {secrets_path}")
        self.hf_token = secrets.get("HF_TOKEN")

        manager = yaml.safe_load(self.manager_config.read_text()) or {}
        control = manager.get("control_plane") or {}
        host = control.get("host", "127.0.0.1")
        self.control_url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{control.get('port', 7014)}"
        self.vllm_port = int((manager.get("worker") or {}).get("vllm_port")
                             or (manager.get("model") or {}).get("port") or 7001)
        state_dir = Path((manager.get("auth") or {}).get("state_dir") or "/var/lib/oumigo")
        self.state_file = state_dir / "runpod" / f"{self.name}.json"


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


# --- RunPod --------------------------------------------------------------------------


def runpod(s: Settings, method: str, path: str, **kwargs) -> httpx.Response:
    headers = {"Authorization": f"Bearer {s.api_key}"}
    try:
        return httpx.request(method, f"{RUNPOD_API}{path}", headers=headers, timeout=60, **kwargs)
    except httpx.RequestError as exc:
        raise Fail(f"RunPod API unreachable: {exc}") from None


def runpod_error(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}: {resp.text[:300]}"
    return f"HTTP {resp.status_code}: {body.get('error') or body}"


DEPLOY = """mutation Deploy($input: PodFindAndDeployOnDemandInput) {
  podFindAndDeployOnDemand(input: $input) { id costPerHr machine { gpuDisplayName location } }
}"""


def create_pod(s: Settings, env: dict[str, str]) -> dict:
    """Create the pod on the first GPU type (and data center) with a free machine.

    Uses RunPod's GraphQL API: unlike the REST API, whose CUDA filter is a list of exact
    versions, it takes a *minimum* CUDA version, so hosts with newer drivers qualify too.
    """
    base = {
        "name": s.name,
        "imageName": s.image,
        "gpuCount": s.gpu_count,
        "cloudType": s.cloud_type,
        "containerDiskInGb": s.container_disk_gb,
        "volumeInGb": 0,
        "ports": "",               # the worker only dials out
        "startSsh": False,
        "env": [{"key": k, "value": v} for k, v in env.items()],
    }
    if s.min_cuda:
        base["minCudaVersion"] = s.min_cuda
    if s.registry_auth_id:
        base["containerRegistryAuthId"] = s.registry_auth_id
    misses = []
    for gpu in s.gpu_types:
        for dc in s.data_centers or [None]:
            body = {**base, "gpuTypeId": gpu, **({"dataCenterId": dc} if dc else {})}
            try:
                resp = httpx.post(RUNPOD_GRAPHQL, json={"query": DEPLOY, "variables": {"input": body}},
                                  headers={"Authorization": f"Bearer {s.api_key}"}, timeout=60)
            except httpx.RequestError as exc:
                raise Fail(f"RunPod API unreachable: {exc}") from None
            data = resp.json() if resp.content else {}
            pod = (data.get("data") or {}).get("podFindAndDeployOnDemand")
            if pod and pod.get("id"):
                return pod
            message = "; ".join(e.get("message", "") for e in data.get("errors") or []) \
                or f"HTTP {resp.status_code}"
            if "available" not in message.lower():  # not a stock miss: a real error
                raise Fail(f"pod not created on {gpu}: {message}")
            misses.append(gpu + (f"@{dc}" if dc else ""))
    raise Fail(f"no machine free for: {', '.join(misses)}")


def get_pod(s: Settings, pod_id: str) -> dict | None:
    resp = runpod(s, "GET", f"/pods/{pod_id}")
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise Fail(f"cannot read pod {pod_id}: {runpod_error(resp)}")
    return resp.json()


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


def cmd_up(s: Settings, wait: bool, wait_timeout: float) -> None:
    state = load_state(s)
    if state and (pod := get_pod(s, state["pod_id"])) and pod.get("desiredStatus") != "TERMINATED":
        raise Fail(f"pod {state['pod_id']} ({s.name}) is already up; run `down` first")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    token_id, token = mint_join_token(s, label=f"runpod {s.name} {stamp}")
    log(f"minted join token {token_id} (ttl {s.token_ttl}, {s.token_uses} uses)")

    env = {"OUMIGO_MANAGER_URL": s.manager_url, "OUMIGO_JOIN_TOKEN": token, **s.extra_env}
    if s.profile:
        env["OUMIGO_MODEL_PROFILE"] = s.profile
    if s.hf_token:
        env["HF_TOKEN"] = s.hf_token
    try:
        pod = create_pod(s, env)
    except Fail:
        oumigo(s, "token", "revoke", token_id)
        log(f"revoked join token {token_id}")
        raise
    node_id = compute_node_id(pod["id"], s.vllm_port)  # the image's host id is the pod id
    save_state(s, {"name": s.name, "pod_id": pod["id"], "node_id": node_id,
                   "join_token_id": token_id, "profile": s.profile,
                   "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    machine = pod.get("machine") or {}
    log(f"created pod {pod['id']} ({s.name}): {machine.get('gpuDisplayName', '?')} in "
        f"{machine.get('location', '?')}, ${pod.get('costPerHr', '?')}/h, expected node {node_id}")
    if wait:
        try:
            wait_serving(s, pod["id"], node_id, wait_timeout)
        except Fail as exc:
            # A pod whose worker never serves only costs money: take it down.
            log(f"{exc.code}; deleting the pod")
            cmd_down(s, pod["id"])
            raise


def wait_serving(s: Settings, pod_id: str, node_id: str, timeout: float) -> None:
    start, last = time.monotonic(), None
    while time.monotonic() - start < timeout:
        worker = manager_worker(s, node_id)
        state = worker.get("state") if worker else "not registered"
        if state != last:
            log(f"worker {node_id}: {state} (+{time.monotonic() - start:.0f}s)")
            last = state
        if state == "serving":
            return
        if state in ("failed", "lost", "stopped"):
            raise Fail(f"worker {node_id} is {state}; see the pod's logs on RunPod")
        if worker is None and (pod := get_pod(s, pod_id)) and pod.get("desiredStatus") != "RUNNING":
            raise Fail(f"pod {pod_id} is {pod.get('desiredStatus')} before the worker joined")
        time.sleep(15)
    raise Fail(f"worker {node_id} not serving after {timeout:.0f}s")


def cmd_status(s: Settings) -> None:
    state = load_state(s)
    if not state:
        print(f"{s.name}: no pod recorded")
        return
    pod = get_pod(s, state["pod_id"])
    if pod is None:
        print(f"{s.name}: pod {state['pod_id']} no longer exists on RunPod")
    else:
        gpu = (pod.get("gpu") or {}).get("displayName", "?")
        print(f"{s.name}: pod {pod['id']} {pod.get('desiredStatus')}, {gpu}, "
              f"${pod.get('costPerHr', '?')}/h, created {state['created_at']}")
    worker = manager_worker(s, state["node_id"])
    if worker:
        print(f"worker {state['node_id']}: {worker.get('state')}, {worker.get('served_name')}, "
              f"{worker.get('transport')}, gpu {(worker.get('capabilities') or {}).get('gpu')}")
    else:
        print(f"worker {state['node_id']}: not registered with the manager")


def cmd_down(s: Settings, pod_id: str | None) -> None:
    state = load_state(s) or {}
    pod_id = pod_id or state.get("pod_id")
    if not pod_id:
        raise Fail(f"no pod recorded for {s.name}; pass --pod-id")
    node_id = state.get("node_id") or compute_node_id(pod_id, s.vllm_port)

    # Revoke first: the router stops routing to it and drops its tunnels within seconds,
    # so deleting the pod cannot cut a client's request mid-stream.
    # Deleting the pod is what stops the bill, so a failed revoke only warns.
    done = oumigo(s, "node", "revoke", node_id)
    log(f"revoked node {node_id}" if done.returncode == 0
        else f"warning: node revoke failed: {done.stderr.strip()}")
    time.sleep(5)
    resp = runpod(s, "DELETE", f"/pods/{pod_id}")
    if resp.status_code not in (200, 204, 404):
        raise Fail(f"pod {pod_id} not deleted ({runpod_error(resp)}); node stays revoked")
    log(f"deleted pod {pod_id}")
    if token_id := state.get("join_token_id"):
        oumigo(s, "token", "revoke", token_id)  # fails harmlessly once the token is spent
    # The pod id is never reused, so the revocation has done its job.
    oumigo(s, "node", "unrevoke", node_id)
    s.state_file.unlink(missing_ok=True)
    log(f"{s.name} is down")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--secrets", type=Path, default=DEFAULT_SECRETS)
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("up", help="create the pod")
    up.add_argument("--wait", action="store_true", help="block until the worker serves")
    up.add_argument("--wait-timeout", type=float, default=1800, help="seconds (default 1800)")
    sub.add_parser("status", help="show the pod and its worker")
    down = sub.add_parser("down", help="delete the pod")
    down.add_argument("--pod-id", help="a pod not recorded in the state file")
    args = parser.parse_args()

    s = Settings(args.config, args.secrets)
    if args.command == "up":
        cmd_up(s, args.wait, args.wait_timeout)
    elif args.command == "status":
        cmd_status(s)
    else:
        cmd_down(s, args.pod_id)


if __name__ == "__main__":
    main()
