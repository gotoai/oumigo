# Workers on rented GPUs (RunPod, Vast.ai)

This guide runs OumiGo workers on rented GPU machines ("pods") that join a manager over
the [tunnel transport](tunnel.md). A pod needs no open ports: the worker dials out to the
manager. The manager setup is in
[deploy-custom-providers-with-tunnel.md](deploy-custom-providers-with-tunnel.md).

Two pieces are involved:

- **The worker image** (`docker/worker/`): vLLM plus OumiGo, nothing deployment-specific.
  Every setting arrives in the pod's environment, so one image serves any fleet and any
  provider.
- **The pod script** (`deploy/pods/oumigo_pods.py`): runs on the manager host, mints a
  join token, rents a machine, waits for the worker, and deletes the machine again. Run
  it by hand or from a timer; the commands are the same.

The script knows three providers, tried in the order you list them:

| Provider | What it rents | Choice within the provider |
|---|---|---|
| `secure.runpod` | RunPod Secure Cloud: RunPod's own data centers | your `runpod.gpu_types`, in order |
| `community.runpod` | RunPod Community Cloud: vetted third-party hosts, cheaper | your `runpod.gpu_types`, in order |
| `vast.ai` | Vast.ai marketplace: verified hosts, filtered by reliability and bandwidth | the cheapest fitting machine |

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

The variables the container reads are listed in `docker/worker/entrypoint.sh`. The pod
script sets these:

| Variable | Purpose |
|---|---|
| `OUMIGO_MANAGER_URL` | the manager's public URL |
| `OUMIGO_JOIN_TOKEN` | a join token minted for this pod |
| `OUMIGO_HOST_ID` | the worker's identity, made up per pod, so a restarted container rejoins as the same node |
| `OUMIGO_MODEL_PROFILE` | the catalog profile to run (see [model-catalog.md](model-catalog.md)) |
| `HF_TOKEN` | read-only Hugging Face token, for gated models |

## 2. Prepare the providers

Set up only the providers you list; one without an API key is skipped.

**RunPod**

1. **API key**: console → Settings → API Keys → create a key with **All** permissions
   (restricted keys cover Serverless endpoints only, not pods).
2. **Registry credential** (private image only): Settings → Container Registry Auth →
   add a user name and a token that may only *read* packages. Note its id, which you can
   also list with `GET https://rest.runpod.io/v1/containerregistryauth`.

**Vast.ai**

1. **Credit**: Vast.ai is prepaid; add credit under Billing.
2. **API key**: console → Keys → create an API key.
3. **Private image**: Vast.ai takes the registry login with each rental, so the script
   needs a read-only registry token (`REGISTRY_TOKEN`) and its user (`vast.registry_user`).

**Hugging Face token**: fine-grained, read-only, from an account that has accepted the
licenses of the gated models you run.

## 3. Set up the manager host

Install the script where the manager's Python can run it, and give it two files:

```bash
sudo install -d /opt/oumigo/pods
sudo install -m 0755 deploy/pods/oumigo_pods.py /opt/oumigo/pods/
sudo install -m 0644 deploy/pods/pods.example.yaml /etc/oumigo/pods.yaml   # then edit
sudo install -m 0600 -o oumigo -g oumigo /dev/null /etc/oumigo/pods.env
sudoedit /etc/oumigo/pods.env
```

`/etc/oumigo/pods.env` holds the secrets, one `KEY=VALUE` per line, with no comment on
the same line as a value. Leave out the keys of providers you do not use:

```
RUNPOD_API_KEY=...
VAST_API_KEY=...
REGISTRY_TOKEN=...
HF_TOKEN=...
```

The script refuses to run if this file is readable by anyone but its owner.

`/etc/oumigo/pods.yaml` holds everything else
([example](../deploy/pods/pods.example.yaml)). The parts that choose the machine:

```yaml
image: ghcr.io/<owner>/oumigo-worker:0.5.1-vllm0.28.0-cu129
disk_gb: 80                      # weights + caches; there is no volume

search:
  providers: [secure.runpod, community.runpod, vast.ai]   # tried in this order
  max_price_per_hour: 0.60       # USD for the whole pod; 0: no bound
  retry_interval: 30             # seconds between searches when nothing fits
  timeout: 1800                  # then give up; 0: search once

runpod:
  gpu_types:                     # tried in this order; list every card the profile fits
    - NVIDIA RTX A6000
    - NVIDIA A40
    - NVIDIA L40
    - NVIDIA RTX 6000 Ada Generation
    - NVIDIA L40S
  registry_auth_id: <id>         # private image only

vast:
  gpu_names: [RTX A6000, A40, L40, L40S, RTX 6000Ada]
  min_reliability: 0.98
  min_download_mbps: 500
  max_download_usd_per_tb: 10
  registry_user: <user>          # private image only, with REGISTRY_TOKEN

worker:
  manager_url: https://oumigo.example.com
  profile: gemma-4-31B-it-qat-w4a16-ct@128k   # must be in the manager's catalog.allow
```

The profile must be allowed in the manager's `catalog:`; see [model-catalog.md](model-catalog.md).

**Price.** The bound is on the pod's hourly price, disk included. RunPod is checked
against its list price before asking, and the new pod's actual price again after: a pod
over the bound is deleted at once. Vast.ai searches within the bound. Two costs fall
outside it:

- **Downloads**: some Vast.ai hosts charge per TB downloaded, and the first start
  downloads about 40 GB (image and weights). `vast.max_download_usd_per_tb` leaves out
  the expensive ones.
- **Speed**: cheap cards are often slow ones. At $0.60 the choice is mostly A6000 / A40
  class; in one test an A40 decoded the 31B profile at 12.6 tokens/s, an RTX 6000 Ada
  at 41.

**Vast.ai's GPU names** differ from RunPod's (`RTX 6000Ada`, not `NVIDIA RTX 6000 Ada
Generation`). Keep the list to cards you trust: the marketplace also offers mining cards
and modified consumer cards that report 48 GB or more.

## 4. Run a pod

Run the script as the manager's service user, with the manager's Python:

```bash
PODS="sudo -u oumigo /opt/oumigo/venv/bin/python /opt/oumigo/pods/oumigo_pods.py"
$PODS up --dry-run     # what each provider would offer now; rents nothing
$PODS up --wait        # rent a machine; return when its worker is serving
$PODS status           # the pod as its provider sees it, the worker as the manager sees it
$PODS down             # delete the pod
```

Options of `up` override `search:` for one run:

```bash
$PODS up --providers vast.ai,secure.runpod --max-price 0.45 --wait
$PODS up --retry-interval 60 --search-timeout 3600 --wait
```

**`up`**

1. Mints a join token (`join_token.ttl`, `join_token.uses`) labelled with the pod name.
2. Asks each provider in turn for an on-demand machine: a listed GPU, a host whose CUDA
   version is at least the image's (from the tag, or `min_cuda_version`), within the
   price bound. The first provider with one wins.
3. If none has one, waits `retry_interval` seconds and asks them all again, for up to
   `timeout` seconds. Then it fails, and the token is revoked.
4. Records the provider, pod id, expected node id and token id (never the token) in
   `<auth state dir>/pods/<name>.json`.
5. With `--wait`, waits until the worker is serving. If the worker fails or does not
   serve within `--wait-timeout` (default 30 minutes), the pod is deleted again: a pod
   that never serves only costs money.

**`down`**

1. Revokes the worker on the manager: the router stops sending it requests and closes its
   tunnels within seconds, so deleting the pod cannot cut a request mid-stream.
2. Deletes the pod. Deleting, not stopping: without a volume a stopped pod keeps nothing
   worth paying for.
3. Revokes the join token, lifts the node revocation (host ids are never reused) and
   forgets the pod.

A pod the state file does not know (say, one left over from a crash) is deleted with
`down --provider <provider> --pod-id <id>`.

**What a pod sees.** The pod's environment, including the join token and `HF_TOKEN`, and
on Vast.ai the registry login, is visible to anyone with access to the provider account,
and to the host's operator on community and marketplace hosts. Keep the join token
short-lived and limited in uses, and the other tokens read-only. Run on community or
marketplace hosts only models and data you would not mind a host operator seeing.

## 5. Run it on a schedule

The same commands run from systemd timers on the manager host. For a pod from 08:40 to
21:00 every day:

```ini
# /etc/systemd/system/oumigo-pod-up.service
[Unit]
Description=Rent a GPU for the OumiGo worker
[Service]
Type=oneshot
User=oumigo
ExecStart=/opt/oumigo/venv/bin/python /opt/oumigo/pods/oumigo_pods.py up --wait

# /etc/systemd/system/oumigo-pod-up.timer
[Timer]
OnCalendar=*-*-* 08:40:00
Persistent=false
[Install]
WantedBy=timers.target
```

Add `oumigo-pod-down.service` / `.timer` the same way with `down` and
`OnCalendar=*-*-* 21:00:00`, then `sudo systemctl enable --now oumigo-pod-up.timer
oumigo-pod-down.timer`. `journalctl -u oumigo-pod-up` shows what each run did. A search
can take up to `search.timeout` and the worker up to `--wait-timeout` more, so the pod
may come up to an hour after the timer fires.
