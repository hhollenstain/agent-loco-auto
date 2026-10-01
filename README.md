# agent-loco

A home-lab coding agent that sits on a project, makes a change, runs that project's tests locally, and only then commits (and optionally publishes) the result. Create-PR also requires a separate review that the diff actually fulfills the stated goal — green tests alone are not enough.

## Install / Quick start

```bash
# Install from source
pip install .
# or with uv
uv sync --extra dev
```

CLI entry point:
```bash
loco --help
```

Test / lint:
```bash
pytest -q
ruff check src tests
```

See the bundled demo:
- `examples/demo-project/`
- Skill conventions: `skills/agent-loco/SKILL.md`

The agent process is the same on an Apple Silicon MacBook and on a Linux box with an NVIDIA GPU. Inference is a separate OpenAI-compatible server:

| Machine | Model server | Why |
| --- | --- | --- |
| MacBook (Apple Silicon) | [Ollama](https://ollama.com) on the host | Docker Desktop cannot pass Metal into a Linux VM |
| Home lab (NVIDIA, including RTX 5090) | Ollama or vLLM in Compose with GPU passthrough | Blackwell needs host driver **570+** and `nvidia-container-toolkit` |

```
  loco run / watch
        │
        ├─ read / write files in one workspace
        ├─ search the web and read public docs
        ├─ run the project's own tests
        ├─ review the diff against the stated goal
        ├─ commit only when tests are green and the goal is met
        └─ optional git push / gh pr
              │
              ▼
     OpenAI-compatible API
     (Ollama Metal · Ollama CUDA · vLLM · cloud)
```

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- `git`
- A model server. On a Mac, install Ollama and pull a coding model:

```bash
brew install ollama
ollama serve
ollama pull qwen2.5-coder:14b
```

Docker is optional. This repo's own tests do not need a model or a GPU.

## Local MacBook

```bash
cd agent-loco
cp .env.example .env
uv sync --extra dev
uv run pytest -q
uv run loco doctor
```

`doctor` tells you whether you are on Apple Silicon or NVIDIA, which backend to use, and whether the model endpoint is reachable. `LOCO_MODEL_NAME` must match a name from `ollama list` (on this Mac that is `Qwen2.5-Coder:14b`, not the library alias).

Point the agent at the bundled demo (it has a failing test on purpose):

```bash
cd examples/demo-project
git init
git add -A
git commit -m "Initial demo project"
cd ../..

uv run loco init examples/demo-project
uv run loco run --workspace examples/demo-project --goal "Make the test suite pass."
```

Continuous mode, using `.loco/goals.md` plus failing tests as the backlog:

```bash
uv run loco watch --workspace /path/to/your/project
```

Attach any local git checkout. The agent will not read or write outside that workspace.

The web UI (`loco ui`) lets you queue cycles, pick an LLM server, choose a model from that host, and switch workspaces. Browse local folders, create a new loco project, or clone a git repo onto this machine so you can run and open PRs locally. Remembered folders live in `.loco/workspaces.json`. Archive a tab to hide it from the session, or remove it from the list; neither deletes the folder on disk. The default LLM server is local Ollama at `http://127.0.0.1:11434/v1`. Change the LLM server field (host:port or a full `/v1` URL) and load models to point at another OpenAI-compatible endpoint. Used servers are remembered in `.loco/servers.json` so they stay selectable after a refresh or restart. `--base-url` on `run`, `watch`, and `ui` does the same from the CLI.

## Onboard a project

```bash
uv run loco init /path/to/your/project
```

That writes:

```text
.loco/config.yaml   # test command, publish settings
.loco/goals.md      # checkbox backlog the watcher consumes
.loco/runs/         # JSON logs written after every cycle
.loco/servers.json  # remembered LLM hosts for the web UI
.loco/workspaces.json  # remembered folders for the workspace picker
```

If `test_command` is omitted, loco infers one (`pytest`, `npm test`, `make test`, `cargo test`, `go test`).

New workspaces enable the bundled process skills: debug, explore, implement,
research, review, security, tdd, and ui. Turn them on or off in the Skills
panel. They are the agent's process; they do not add extra tools.

Publish/create-PR is off until you turn it on in `.loco/config.yaml` or pass `--create-pr`. That option opens a feature branch and pull request; it never pushes to `main`. Commits still require a green test run when `LOCO_REQUIRE_TESTS=true`.

## Containerized

The agent image is multi-arch (`linux/arm64` and `linux/amd64`) and does **not** need a GPU. Mount a host directory at `/workspaces` and talk to a model server.

`LOCO_WORKSPACE` (default `./workspaces`) is the only project volume. Cloned repos and each project's `.loco/` (config, goals, run history) persist on the host because they live *inside* that folder. Do not bind-mount this app's `.loco` over `/workspaces/.loco`; that hides the mounted project's own config.

`docker compose up` starts the web UI on [http://127.0.0.1:8080](http://127.0.0.1:8080). One-off commands replace that with `run`, `clone`, `init`, `watch`, or `doctor`.

### Clone a repo into the container

`git` is in the image; the entrypoint is `loco`, so clone with `loco clone` (or clone on the host into `$LOCO_WORKSPACE`). The destination must not exist yet — clone first, then `init` only if you still need scaffolding.

```bash
mkdir -p workspaces
export LOCO_WORKSPACE="$(pwd)/workspaces"
docker compose run --rm agent clone git@github.com:user/repo.git
docker compose run --rm agent init /workspaces/repo
docker compose run --rm agent run -w /workspaces/repo --goal "Add a README.md"
```

Private GitHub repos need the SSH mount in `docker-compose.yml` uncommented, or clone on the host:

```bash
git clone git@github.com:user/repo.git "$LOCO_WORKSPACE/repo"
```

### Mac + Docker Desktop

Keep Ollama on the host (Metal), then:

```bash
export LOCO_WORKSPACE="$(pwd)/workspaces"
docker compose -f docker-compose.yml -f docker-compose.mac.yml run --rm agent doctor
docker compose -f docker-compose.yml -f docker-compose.mac.yml up --build
# UI: http://127.0.0.1:8080
docker compose -f docker-compose.yml -f docker-compose.mac.yml run --rm agent \
  run --workspace /workspaces --goal "Make the test suite pass."
```

### NVIDIA home lab (5090 and similar)

On the Linux host:

1. NVIDIA driver **570 or newer** (Blackwell / RTX 50-series reports compute capability 12.0).
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
3. `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

Then:

```bash
export LOCO_WORKSPACE="$(pwd)/workspaces"
export LOCO_MODEL_NAME=qwen2.5-coder:32b
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up --build -d ollama
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml exec ollama ollama pull qwen2.5-coder:32b
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm agent doctor
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm agent \
  clone git@github.com:user/repo.git
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm agent \
  watch --workspace /workspaces/repo
```

A 32B coder model is a reasonable default on a 32 GB 5090. Swap `LOCO_MODEL_NAME` if you prefer vLLM or a larger quant. Any server that speaks `/v1/chat/completions` works; set `LOCO_MODEL_BASE_URL` accordingly.

## Safety

- All file and shell tools are rooted in `--workspace`. Path escape is rejected.
- `web_search` and `fetch_url` only reach public http(s) URLs. Private,
  loopback, and link-local addresses are rejected.
- Likely secrets (`.env`, keys, `credentials.json`) cannot be committed.
- Force-push and `--no-verify` are not available.
- Auto-commit is skipped when tests fail and `LOCO_REQUIRE_TESTS` is on.
- Create-PR never pushes to `main`/`master`. It opens a `loco/*` branch and a pull request instead.

This is still a coding agent with a shell inside a trusted workspace. Do not point it at a tree you would not edit yourself.

## CLI

| Command | Purpose |
| --- | --- |
| `loco doctor` | Hardware, git/docker, model health |
| `loco init [path]` | Write `.loco/` scaffolding (path must already exist) |
| `loco clone URL [DIR]` | `git clone` into `DIR` or `./<repo>` and write `.loco/` |
| `loco run -w PATH -g "..." -m MODEL --base-url URL` | One improve → test → commit cycle |
| `loco ui -w PATH --base-url URL` | Local web UI to queue and run tasks (`--web-ui` on `run` also works) |
| `loco watch -w PATH` | Repeat cycles on an interval |

Environment variables are listed in `.env.example`.
