# Model catalog: several models in one fleet

Since 0.5.0, one OumiGo fleet can serve several models at once. Each worker runs the model
that fits its GPU; clients choose a model by id, and the router sends each request to a
worker running that model. Since 0.5.5, a model can only come from a catalog profile.

## The idea: one start-time configuration is one model

vLLM fixes many settings when it starts: context length (`--max-model-len`), memory share,
concurrency, tensor parallelism, quantization flags, parsers. Changing any of them means
restarting the backend. So two workers running the same weights with different start-time
settings are, for every practical purpose, **two different models**.

OumiGo calls one such configuration a **profile**. A profile's **id is the model id**:

- clients put it in `"model"`,
- `GET /v1/models` lists it,
- responses carry it,
- the backend serves under it (`--served-model-name`).

Ids look like `<repo-name>@<context>[-<variant>]`:

| Id | Meaning |
|---|---|
| `gemma-4-12B-it-qat-w4a16-ct@128k` | `google/gemma-4-12B-it-qat-w4a16-ct`, 128k context, defaults otherwise |
| `Qwen3.8-27B-W8A16@128k-c10` | 128k context, 10 concurrent requests |
| `Qwen3.5-9B@32k-bnb4` | 32k context, quantized in flight with bitsandbytes 4-bit |
| `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@256k-c8` | 256k context, 8 concurrent requests |

A vendor repo name such as `google/gemma-4-12B-it-qat-w4a16-ct` describes weights, not a
running configuration, so it is **not** a model id.

## Profiles

A profile is a YAML file named `<id>.yaml`:

```yaml
id: gemma-4-31B-it-qat-w4a16-ct@128k        # must equal the file name
model: google/gemma-4-31B-it-qat-w4a16-ct   # the weights: a Hugging Face repo
description: Gemma 4 31B instruct, Google QAT W4A16, 128k context, one request at a time
engine:                                     # start-time settings → `vllm serve`
  max_model_len: 131072
  gpu_memory_utilization: 0.90
  dtype: auto
  tensor_parallel_size: 1
  max_concurrent_requests: 1                # the router's per-worker admission cap
  extra_args:                               # verbatim extra `vllm serve` flags
    - --enable-auto-tool-choice
    - --tool-call-parser
    - gemma4
    - --reasoning-parser
    - gemma4
caps:                                       # enforced by the router, never sent to vLLM
  max_output_tokens: 4096                   # ceiling on max_tokens (also when none is sent)
  max_audio_seconds: null                   # total audio per request; null = no cap
  max_queue_seconds: 1800                   # wait for a free slot; null = router default (300)
agent:                                      # client timeouts served at /agent-defaults
  turn_timeout: 1800.0
  stall_timeout: 600.0
requires:                                   # checked by the worker before it loads anything
  min_vram_gib: 44                          # total VRAM of GPU 0
  vllm: ">=0.28"                            # version specifier
tested: RTX A6000 48 GB, vLLM 0.28.0        # informational
```

Only `id` and `model` are required. Unknown keys are rejected, so a typo stops the manager
at startup instead of reaching a worker minutes later. `agent:` may be left out; the
manager's own `agent:` block then applies.

A profile's `task` says what its model is for: `chat` (the default) or `embed` for an
embedding model, which may also carry an `embed:` block with its vector size and its
query/document prompts. See [embeddings.md](embeddings.md).

`requires` turns a slow failure into a fast one: a worker whose GPU is too small exits with
"profile … needs a GPU with at least 44 GiB; GPU 0 has 15.6 GiB" before it downloads or
loads anything.

### Shipped profiles

OumiGo ships the profiles its maintainers have run, in `src/oumigo/catalog/<vendor>/`:

| Id | Weights |
|---|---|
| `gemma-4-12B-it-qat-w4a16-ct@128k` | `google/gemma-4-12B-it-qat-w4a16-ct` |
| `gemma-4-31B-it-qat-w4a16-ct@128k` | `google/gemma-4-31B-it-qat-w4a16-ct` |
| `Muse-Glimmer-30B-W8A16@128k-c10` | `GotoAI-Inc/Muse-Glimmer-30B-W8A16` |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W8A16@128k-c10` | `GotoAI-Inc/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W8A16` |
| `Qwen3.8-27B-W8A16@128k-c10` | `GotoAI-Inc/Qwen3.8-27B-W8A16` |
| `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@256k-c8` | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` |
| `Qwen3.5-9B@32k-bnb4` | `Qwen/Qwen3.5-9B` |
| `ruri-v3-310m@8k` (embedding) | `cl-nagoya/ruri-v3-310m` |
| `harrier-oss-v1-0.6b@8k` (embedding) | `microsoft/harrier-oss-v1-0.6b` |
| `harrier-oss-v1-0.6b@32k` (embedding) | `microsoft/harrier-oss-v1-0.6b` |

Each file's `tested:` line says what it was verified on. Untested templates to adapt live
in [`config_examples/catalog/`](../config_examples/catalog/); they are not loaded
automatically.

### Your own profiles

Put profile files in a directory and list it in `catalog.dirs`. A profile there with the
same id as a shipped one replaces it (the manager logs which). Two files with the same id
in one directory are an error.

To run a shipped model with a different context or concurrency, copy its file, change the
values, and give it the id that says so, e.g. `gemma-4-12B-it-qat-w4a16-ct@32k-c4`.

## The fleet: `catalog:` in manager.yaml

```yaml
catalog:
  dirs: [/etc/oumigo/catalog]        # optional; shipped profiles are always available
  allow:                             # the profiles workers may run in this fleet
    - gemma-4-12B-it-qat-w4a16-ct@128k
    - gemma-4-31B-it-qat-w4a16-ct@128k
  default: gemma-4-12B-it-qat-w4a16-ct@128k   # for workers that choose nothing

worker:
  vllm_port: 7001                    # backend port workers prefer

agent:                               # fallback client timeouts for profiles without agent:
  turn_timeout: 90.0
  stall_timeout: 120.0
```

The rules:

1. **Candidates.** The shipped profiles plus those in `dirs` are the only models a fleet
   can ever run.
2. **Allow list.** `allow` picks this fleet's profiles from the candidates. It must name at
   least one, and every name must exist.
3. **Default.** `default` is optional and must be in `allow`. Without one, every worker
   has to choose.

The `catalog:` block is **required**. A manager without one doesn't start, and shows a
minimal example. A missing block does not mean "every shipped profile": that set would
change with each package upgrade. The 0.4-style `model:` block was removed in 0.5.5;
the manager refuses it and asks for a profile in `allow` instead. The manager logs the
allowed profiles and the default at startup.

## Workers: choosing a profile

```bash
OUMIGO_MODEL_PROFILE=gemma-4-31B-it-qat-w4a16-ct@128k
```

The worker resolves its profile in this order:

1. `OUMIGO_MODEL_PROFILE` is set and allowed: the worker runs it. If the fleet doesn't
   allow it, the worker exits with the list of allowed ids. It never falls back to the
   default: running a model you didn't ask for is worse than stopping.
2. It is unset: the worker runs the fleet's default.
3. Nor is there a default: the worker exits with `Model profile cannot be resolved: this
   fleet has no default; set OUMIGO_MODEL_PROFILE to one of: ...`.

The manager makes this decision, not the worker. The worker asks the manager for the
profile's settings before it registers, and the manager checks the profile again at
registration. **The worker carries no model settings of its own**: a container image needs
only the profile id in its environment, and the manager's copy of a profile always wins
over the worker's package.

Settings that describe the **node**, not the model, still come from the worker's
environment:

| Variable | Purpose |
|---|---|
| `MODEL_STORAGE_LOCATION` | Weights on this node's disk (`file:///path`), or `none` to download despite a fleet-wide location |
| `HF_HOME`, `HF_TOKEN` | Hugging Face cache and token (gated models) |

`MODEL_NAME` and `MAX_MODEL_LEN` were removed in 0.5.5. A worker with either set exits at
once, saying to use `OUMIGO_MODEL_PROFILE`; it doesn't ignore them, because an old `.env`
would otherwise start the default model while its owner believes they chose another. For
a different context length, choose or create a profile with that context.

## Clients: choosing a model

Send a model id, or `any-model`:

```python
from openai import OpenAI

client = OpenAI(base_url="https://oumigo.example.com/v1", api_key="oumigo-key-...")
print([m.id for m in client.models.list()])     # any-model + the ids being served
resp = client.chat.completions.create(
    model="gemma-4-31B-it-qat-w4a16-ct@128k",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.model)                               # the id that answered
```

With the OumiGo Python client: `manager.create_agent(model="gemma-4-31B-it-qat-w4a16-ct@128k")`,
or leave `model` out for `any-model`.

| Request `model` | Goes to |
|---|---|
| a model id | workers running that profile |
| `any-model`, or no `model` field | any serving chat worker; the response names the model that answered |
| anything else, including a vendor repo name | nothing: `404 model_not_found` |
| an embedding model, on the chat paths | nothing: `400 wrong_task` (use `/v1/embeddings`) |

When nothing can take the request right now, the router answers `503` with a `code` that
says why:

| `code` | When | Message (shown to end users) | `Retry-After` |
|---|---|---|---|
| `model_loading` | a worker for the model is still starting | `<model> is starting, retry shortly.` | 30 |
| `model_unavailable` | no worker runs the model (or none responds) | `<model> is not available right now.` | — |
| `model_busy` | its workers stayed busy past `caps.max_queue_seconds` | `<model> is busy, retry shortly.` | 30 |

For `any-model` the messages read "A model is starting", "No model is available", "All
models are busy". They are written for end users, since client apps show them as they
are; what the router saw (which workers, in which state) goes to the manager's log.

All errors use OpenAI's error format, so SDKs raise them with their message. The OpenAI
SDKs also retry a `503` on their own, waiting `Retry-After` between attempts:

```json
{"error": {"message": "Model 'gemma-x' is not available. Available models: gemma-4-12B-it-qat-w4a16-ct@128k, gemma-4-31B-it-qat-w4a16-ct@128k, any-model.",
           "type": "invalid_request_error", "param": "model", "code": "model_not_found"}}
```

### `GET /v1/models`

The router answers it from its live view of the workers: `any-model` first, then each id
with at least one serving worker. An allowed model with no serving worker is left out, so
apps that fill a menu from this list never offer a model that would only time out.
`any-model` is listed only while at least one chat model is serving, since it only ever
reaches chat workers; a fleet of embedding workers alone does not list it.

```json
{"object": "list", "data": [
  {"id": "any-model", "object": "model", "owned_by": "oumigo", "oumigo": {"task": "chat"}},
  {"id": "gemma-4-12B-it-qat-w4a16-ct@128k", "object": "model", "owned_by": "google",
   "oumigo": {"weights": "google/gemma-4-12B-it-qat-w4a16-ct", "task": "chat",
              "max_model_len": 131072, "serving": 1}}
]}
```

Embedding models also list their `dimensions` and `input_types`; see
[embeddings.md](embeddings.md).

`GET /agent-defaults?model=<id>` returns that model's client timeouts. For `any-model` (or
no `model`), it returns the **longest** timeouts among the chat models being served, since
any of them may answer; "no limit" outranks every number.

## Queueing

Each worker accepts up to its own profile's `max_concurrent_requests` (one request at a
time on the transformer backend). When every worker for a
model is busy, its requests wait in the router's queue. A slot that frees goes to the
oldest waiting request that can use it, so a busy model never holds up requests for
another model, and order is kept within each model.

A request waits at most its model's `caps.max_queue_seconds` (default 300 s), then gets
`503`. Raise it for a model whose requests are long, such as one 128k-context sequence at a
time, and raise the model's `agent:` timeouts with it so clients wait as long. An
`any-model` request waits as long as the longest limit among the chat models being served.

## Upgrading to 0.5.5

| What | Change |
|---|---|
| `manager.yaml` with a `model:` block | **Refused.** Put the model in a profile and list it in `catalog.allow` |
| `manager.yaml` without `catalog:` (an open fleet) | **Refused.** Add a `catalog:` block |
| Workers using `MODEL_NAME` | **Refused.** Set `OUMIGO_MODEL_PROFILE` to a profile id |
| Workers using `MAX_MODEL_LEN` | **Refused.** Choose or create a profile with that context |
| Python API: `oumigo_get_or_create_manager(model=...)` | Removed. Pass `catalog=` or a `config_file` with `catalog:` |
| Python API: `oumigo_create_worker(model_name=...)` | Removed. Pass `profile=` |
| Per-worker capacity | Each worker now gets its own profile's `max_concurrent_requests`. Before, every vLLM worker got the default profile's value |

Upgrade the manager and the workers together. A 0.5.5 worker exits when a manager hands it
no profile, saying the manager needs upgrading; a 0.5.5 manager refuses a registration
without a profile.
