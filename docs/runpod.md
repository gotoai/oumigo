# Workers on RunPod

This guide runs OumiGo workers as RunPod GPU pods that join a manager over the
[tunnel transport](tunnel.md). A pod needs no open ports: the worker dials out to the
manager. The manager setup is in
[deploy-custom-providers-with-tunnel.md](deploy-custom-providers-with-tunnel.md).

Two pieces are involved:

- **The worker image** (`docker/worker/`): vLLM plus OumiGo, nothing deployment-specific.
  Every setting arrives in the pod's environment, so one image serves any fleet.
- **The pod script** (`deploy/runpod/oumigo_runpod.py`): runs on the manager host, mints
  a join token, creates the pod, waits for the worker, and deletes the pod again. Run it
  by hand or from a timer; the commands are the same.

## 1. Build and push the image

```bash
docker build --build-arg VLLM_VERSION=0.28.0 --build-arg OUMIGO_VERSION=0.5.1 \
    --build-arg VLLM_VARIANT=-cu129 \
    -t ghcr.io/<owner>/oumigo-worker:0.5.1-vllm0.28.0-cu129 docker/worker
docker push ghcr.io/<owner>/oumigo-worker:0.5.1-vllm0.28.0-cu129
```

The image starts from `vllm/vllm-openai`, adds the packages that image lacks (vLLM's
audio extra, `bitsandbytes`) without changing any it ships, and installs `oumigo` from
PyPI. Weights are never baked in; each pod downloads them on start.

**Pick the CUDA variant by the hosts you want to reach.** A container needs a host driver
at least as new as its CUDA version:

| `VLLM_VARIANT` | Image CUDA | Host driver | Runs on |
|---|---|---|---|
| (empty) | vLLM's default (13.0 for vLLM 0.28) | 580+ | CUDA 13.0+ hosts |
| `-cu129` | 12.9 | 575+ | CUDA 12.9+ hosts: a larger pool |

Put the variant in the tag (`-cu129`): the pod script reads the CUDA floor from it.

The variables the container reads are listed in `docker/worker/entrypoint.sh`. The ones
that matter here:

| Variable | Set by | Purpose |
|---|---|---|
| `OUMIGO_MANAGER_URL` | pod script | the manager's public URL |
| `OUMIGO_JOIN_TOKEN` | pod script | a join token minted for this pod |
| `OUMIGO_MODEL_PROFILE` | pod script | the catalog profile to run (see [model-catalog.md](model-catalog.md)) |
| `HF_TOKEN` | pod script | read-only Hugging Face token, for gated models |
| `OUMIGO_HOST_ID` | image | defaults to `$RUNPOD_POD_ID`, the worker's stable identity |

## 2. Prepare RunPod

1. **API key**: RunPod console → Settings → API Keys → create a key with **All**
   permissions (restricted keys cover Serverless endpoints only, not pods).
2. **Registry credential** (private image only): Settings → Container Registry Auth →
   add a user name and a token that may only *read* packages. Note its id, which you can
   also list with `GET https://rest.runpod.io/v1/containerregistryauth`.
3. **Hugging Face token**: fine-grained, read-only, from an account that has accepted the
   licenses of the gated models you run.

## 3. Set up the manager host

Install the script where the manager's Python can run it, and give it two files:

```bash
sudo install -d /opt/oumigo/runpod
sudo install -m 0755 deploy/runpod/oumigo_runpod.py /opt/oumigo/runpod/
sudo install -m 0644 deploy/runpod/runpod.example.yaml /etc/oumigo/runpod.yaml   # then edit
sudo install -m 0600 -o oumigo -g oumigo /dev/null /etc/oumigo/runpod.env
sudoedit /etc/oumigo/runpod.env
```

`/etc/oumigo/runpod.env` holds the secrets, one `KEY=VALUE` per line, with no comment on
the same line as a value:

```
RUNPOD_API_KEY=...
HF_TOKEN=...
```

The script refuses to run if this file is readable by anyone but its owner.

`/etc/oumigo/runpod.yaml` holds everything else
([example](../deploy/runpod/runpod.example.yaml)):

```yaml
pod:
  name: oumigo-48gb
  image: ghcr.io/<owner>/oumigo-worker:0.5.1-vllm0.28.0-cu129
  gpu_types:                     # tried in this order; list every card the profile fits
    - NVIDIA RTX A6000
    - NVIDIA A40
    - NVIDIA L40
    - NVIDIA RTX 6000 Ada Generation
    - NVIDIA L40S
  cloud_type: SECURE
  container_disk_gb: 80          # weights + caches; there is no volume
  registry_auth_id: <id>         # private image only
worker:
  manager_url: https://oumigo.example.com
  profile: gemma-4-31B-it-qat-w4a16-ct@128k   # must be in the manager's catalog.allow
join_token:
  ttl: 13h
  uses: 3
```

The profile must be allowed in the manager's `catalog:`; see [model-catalog.md](model-catalog.md).

## 4. Run a pod

Run the script as the manager's service user, with the manager's Python:

```bash
RUNPOD="sudo -u oumigo /opt/oumigo/venv/bin/python /opt/oumigo/runpod/oumigo_runpod.py"
$RUNPOD up --wait      # create the pod; return when its worker is serving
$RUNPOD status         # the pod as RunPod sees it, the worker as the manager sees it
$RUNPOD down           # delete the pod
```

**`up`**

1. Mints a join token (`join_token.ttl`, `join_token.uses`) labelled with the pod name.
2. Creates an on-demand pod on the first GPU type in `gpu_types` with a free machine,
   restricted to hosts whose CUDA version is at least the image's (from the tag, or
   `pod.min_cuda_version`). RunPod stock changes by the minute, so list every card the
   profile fits.
3. Records the pod id, the expected node id and the token id (never the token) in
   `<auth state dir>/runpod/<name>.json`.
4. With `--wait`, waits until the worker is serving. If the worker fails or does not
   serve within `--wait-timeout` (default 30 minutes), the pod is deleted again: a pod
   that never serves only costs money.

If no machine is free, nothing is created and the token is revoked.

**`down`**

1. Revokes the worker on the manager: the router stops sending it requests and closes its
   tunnels within seconds, so deleting the pod cannot cut a request mid-stream.
2. Deletes the pod. Deleting, not stopping: without a volume a stopped pod keeps nothing
   worth paying for.
3. Revokes the join token, lifts the node revocation (pod ids are never reused) and
   forgets the pod.

**What a pod sees.** The pod's environment, including the join token and `HF_TOKEN`, is
visible to anyone with access to the RunPod account. Keep the join token short-lived and
limited in uses, and the Hugging Face token read-only.

## 5. Run it on a schedule

The same commands run from systemd timers on the manager host. For a pod from 08:40 to
21:00 every day:

```ini
# /etc/systemd/system/oumigo-runpod-up.service
[Unit]
Description=Create the OumiGo RunPod worker
[Service]
Type=oneshot
User=oumigo
ExecStart=/opt/oumigo/venv/bin/python /opt/oumigo/runpod/oumigo_runpod.py up --wait

# /etc/systemd/system/oumigo-runpod-up.timer
[Timer]
OnCalendar=*-*-* 08:40:00
Persistent=false
[Install]
WantedBy=timers.target
```

Add `oumigo-runpod-down.service` / `.timer` the same way with `down` and
`OnCalendar=*-*-* 21:00:00`, then `sudo systemctl enable --now oumigo-runpod-up.timer
oumigo-runpod-down.timer`. `journalctl -u oumigo-runpod-up` shows what each run did.
