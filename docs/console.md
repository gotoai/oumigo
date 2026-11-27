# The web console

The manager runs a small web console next to the control plane: a **Workers** page, where
you can close a worker gracefully, and charts of **GPU Utilization** and **Token
Generation**. It runs as its own
process, so a problem in it never affects inference or heartbeats.

- [Reaching it](#reaching-it)
- [Signing in](#signing-in)
- [Workers and Close](#workers-and-close)
- [Closing from the command line](#closing-from-the-command-line)
- [GPU Utilization](#gpu-utilization)
- [Token Generation](#token-generation)
- [Drain limit](#drain-limit)
- [Exit codes](#exit-codes)

## Reaching it

The console binds **loopback only** (default `127.0.0.1:7080`; `dashboard.enabled: false`
turns it off). Reach it through an SSH tunnel, also from another machine on your LAN:

```yaml
dashboard:
  host: 127.0.0.1   # a loopback address; any other is refused at startup
  port: 7080
```

```bash
ssh -L 7080:127.0.0.1:7080 user@oumigo.example.com
# then open http://localhost:7080
```

A `dashboard.host` that isn't a loopback address stops the manager at startup with a
message. The console answers only requests addressed to `localhost` / `127.0.0.1`, so a
website open in your browser can't reach it through DNS tricks while the tunnel is up.
Every action that changes something needs the page's CSRF token and a same-origin
request.

## Signing in

One admin account, set in the manager's environment or `.env` file:

| Variable | Value |
|---|---|
| `MG_ADMIN_USER` | the login name: 3 to 40 lowercase letters, digits, `.`, `_` or `-` |
| `MG_ADMIN_PASSWORD_HASH` | the hash printed by `oumigo manager admin-password` |

**Only a signed-in admin can close workers.** With the account set, every page needs
sign-in. If either variable is missing or empty there is no admin: the console is
view-only (Close is disabled) and the top bar offers **Create Admin Account**.

`oumigo manager serve` and `oumigo manager run` read `.env` from the working directory
(`--env-file` for another path); variables already in the environment win over the file.
With the systemd unit in [deploy-custom-providers-with-tunnel.md](deploy-custom-providers-with-tunnel.md)
the file is `/var/lib/oumigo/.env`. Keep it mode 0600 and don't put comments on the same
line as a value.

### Create the account

**In the console.** While there is no admin, the top bar shows **Create Admin Account**.
The page asks for a login name and a password (twice), writes both variables into the
manager's `.env` (other lines are kept) and turns sign-in on at once. When it can't write
the file, or a variable is set in the manager's environment rather than the file, the page
shows the manual steps instead. Create the account soon after deploying: until then,
anyone who can reach the console (through SSH) could create it first.

**By hand.**

```bash
oumigo manager admin-password          # asks for the password twice, prints the hash line
```

Add the printed `MG_ADMIN_PASSWORD_HASH=…` line and `MG_ADMIN_USER=<login>` to the
`.env` file, then restart the manager. Workers keep serving across a manager restart.

The password needs at least 10 printable ASCII characters (no spaces), with a lowercase
letter, an uppercase letter, a digit and a symbol. The hash is scrypt, written as
`scrypt:<n>:<r>:<p>:<salt>:<key>` (no characters that need quoting).

### Sessions

A session ends after 2 hours without a page load or action, on sign-out, and when the
password hash changes. Five wrong passwords in a row lock sign-in for 15 minutes.

**Forgotten password:** make a new hash with `oumigo manager admin-password`, replace the
line in `.env` and restart the manager. Or empty `MG_ADMIN_PASSWORD_HASH`, restart, and
set a new password on the Create Admin Account page through the tunnel.

## Workers and Close

The Workers page lists the registered workers with the same columns as the manager
console's `workers` command, refreshed every 5 seconds.

**Close** drains and stops one worker:

1. The manager stops sending it new requests at once.
2. With its next heartbeat (within `heartbeat.interval_s`, 10 s by default) the worker
   is told to stop. It finishes the requests in progress, waiting at most its
   [drain limit](#drain-limit), then stops its backend and exits.
3. The row shows `draining`, then `stopped`.

Close acts on one worker only. In a container that runs several profiles
([model-catalog.md](model-catalog.md)), the other workers keep serving; the container's
main process exits once all of them have closed. What happens to the container or
machine after that is up to its platform: one that restarts exited containers starts a
new run of the worker, which joins as usual.

Close is offered for `registering`, `initializing` and `serving` workers. The request is
kept in the manager's state directory until the worker reports `stopped`, so a manager
restart in the middle of a close continues it.

## Closing from the command line

On the manager host:

```bash
oumigo manager node close Worker#3 -c /etc/oumigo/manager.yaml
oumigo manager node close 0123456789abcdef -c /etc/oumigo/manager.yaml   # or the node id
```

This writes the same request the console writes; the running manager acts on it within
seconds.

## GPU Utilization

Four charts of each worker's GPU utilization (the mean across its GPUs), times in UTC:

| Chart | Grid | Each point |
|---|---|---|
| Past 60 Minutes | 5 s | 6-sample moving average |
| Past 48 Hours | 5 min | mean of the 5 s samples in those 5 minutes |
| Past 10 Days | 30 min | mean of the samples in those 30 minutes |
| Past 45 Days | 2 h | mean of the samples in those 2 hours |

Each chart is a rolling window: the newest point, on the right, covers the period so far
and keeps updating; the oldest drops off the left. A worker that reported nothing in a
period leaves a gap. Lines are labeled with the worker's current `Worker#N`, which the
manager assigns again after a restart.

The 60-minute chart reads the latest samples. The longer ones read averages the console
keeps in `console_rollups.db` in the manager's state directory, so they survive restarts.
They start empty and fill from the moment the console first runs.

## Token Generation

Two views with the same four ranges and grids as GPU Utilization, from the token counters
each worker's vLLM reports: **generated** tokens (solid lines) and **input** (prompt)
tokens (dashed). Embedding models only count input tokens. Each worker gets a line per
kind, plus an **All workers** total per kind (the thick line). MT = million tokens.

| View | Unit | Each point |
|---|---|---|
| Token Generation | MT / min | the tokens of that 5 s slot or bucket, divided by its length; the Past 60 Minutes chart adds the 6-sample moving average |
| Token Generation (Cumulative) | MT | the running total from the chart's left edge |

The newest point of the per-minute charts divides the tokens so far by the time elapsed in
its bucket. vLLM's counters restart at 0 when its engine restarts; a drop is counted as a
restart, never as negative tokens. Samples a worker could not deliver (see
`WK_METRICS_MAX_KEEPING_AGE`) lose no tokens: the next sample carries the total.

## Drain limit

How long a stopping worker waits for in-flight requests. It stops as soon as nothing is
in flight, so the limit only matters for long requests. The worker takes the first of:

1. `WK_DRAIN_TIMEOUT` (seconds) in the worker's environment or `.env`;
2. its profile's `agent.turn_timeout`, or the manager's `agent.turn_timeout` for a profile
   without an `agent:` block (a client stops waiting for a turn after that long anyway);
3. 120 seconds.

The same limit applies when the worker is stopped by a signal (`docker stop`,
`systemctl stop`, Ctrl-C). Give the container or unit a stop timeout longer than the
drain limit plus 30 seconds, e.g. `docker stop -t 1860` for a 1800 s limit; otherwise the
platform kills the worker in the middle of the drain.

## Exit codes

`oumigo worker run` exits with:

| Code | Meaning |
|---|---|
| 0 | stopped normally: closed by the manager, or stopped by a signal |
| 1 | failed: the backend kept crashing, no free port, or a startup error |
| 3 | refused by the manager (the node was revoked) |

So `docker run --restart on-failure` restarts a failed worker but not a closed one.
