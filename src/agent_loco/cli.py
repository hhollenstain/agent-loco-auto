from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from ruamel.yaml import YAML

from agent_loco import __version__
from agent_loco.config import Settings
from agent_loco.hardware import command_available, detect_hardware
from agent_loco.llm.client import (
    OpenAICompatClient,
    check_model_endpoint,
    normalize_model_base_url,
)
from agent_loco.logging import setup_logging
from agent_loco.runtime.improve import run_cycle
from agent_loco.runtime.project import write_default_project_files
from agent_loco.runtime.watch import watch as watch_loop

REQUIRED_ENV_VARS = ["LOCO_MODEL_NAME", "LOCO_MODEL_BASE_URL"]


app = typer.Typer(
    name="loco",
    help="Home-lab coding agent: improve a project, test locally, commit when green.",
    no_args_is_help=True,
)
console = Console()


def _load_config_file() -> None:
    """Load environment variables from a config file if it exists.
    
    Supports .env, config.yaml, and config.yml in the current directory.
    """
    yaml = YAML()
    config_paths = [
        Path.cwd() / ".env",
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                if config_path.name == ".env":
                    with open(config_path) as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                key, _, value = line.partition("=")
                                os.environ[key.strip()] = value.strip()
                else:
                    with open(config_path) as f:
                        data = yaml.load(f)
                        if isinstance(data, dict):
                            for key, value in data.items():
                                if isinstance(value, (str, int, float, bool)):
                                    os.environ.setdefault(f"LOCO_{key.upper()}", str(value))
                console.print(f"[green]Loaded config from {config_path}[/green]")
                return
            except Exception:
                console.print(f"[yellow]Warning: Could not load config from {config_path}[/yellow]")


def _validate_env(require: bool = True) -> None:
    """Check required environment variables and exit with a helpful message if missing.
    
    Args:
        require: If False, skip validation to allow UI mode without env vars.
    """
    if not require:
        return
    
    # Try to load config file first
    _load_config_file()
    
    missing = [var for var in REQUIRED_ENV_VARS if var not in os.environ]
    if missing:
        console.print("[red]ERROR:[/red] Required environment variables not set:")
        for var in missing:
            console.print(f"  - {var}")
        console.print("\nPlease create a .env file from .env.example or export these variables.")
        console.print("\nYou may also create config.yaml or config.yml in the current directory.")
        raise SystemExit(1)


def _settings(**overrides: object) -> Settings:
    """Create settings with optional overrides.
    
    Args:
        **overrides: Settings to override. Can include model_name, model_base_url,
            create_pr, auto_commit, and any other Settings field.
    
    Returns:
        A Settings instance with the overrides applied.
    """
    settings = Settings()
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)
    settings.model_base_url = normalize_model_base_url(settings.model_base_url)
    setup_logging(settings.log_level)
    return settings


def _llm(settings: Settings) -> OpenAICompatClient:
    return OpenAICompatClient(
        model=settings.model_name,
        base_url=settings.model_base_url,
        api_key=settings.model_api_key,
    )


def _print_version(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the loco version and exit.",
            callback=_print_version,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Home-lab coding agent."""
    pass


@app.command()
def doctor() -> None:
    """Report hardware, tooling, and model-endpoint health."""
    _load_config_file()
    _validate_env()
    settings = _settings()
    hw = detect_hardware()

    table = Table(title="agent-loco doctor")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("os", f"{hw.os_name}/{hw.arch}")
    table.add_row("apple silicon", "yes" if hw.is_apple_silicon else "no")
    table.add_row("nvidia", hw.gpu_name or "no")
    table.add_row("recommended backend", hw.recommended_backend)
    table.add_row("recommended model", hw.recommended_model)
    table.add_row("git", "yes" if command_available("git") else "missing")
    table.add_row("docker", "yes" if command_available("docker") else "missing (optional)")
    table.add_row("gh", "yes" if command_available("gh") else "missing (needed for PRs)")
    table.add_row("model url", settings.model_base_url)
    table.add_row("model name", settings.model_name)

    ok, detail = check_model_endpoint(settings.model_base_url, settings.model_api_key)
    table.add_row("model endpoint", detail if ok else f"DOWN — {detail}")
    console.print(table)
    for note in hw.notes:
        console.print(f"[yellow]{note}[/yellow]")
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def init(
    workspace: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, resolve_path=True),
    ] = Path("."),
) -> None:
    """Create .loco/config.yaml and .loco/goals.md in a project."""
    _validate_env()
    created = write_default_project_files(workspace)
    if created:
        for path in created:
            console.print(f"created {path}")
    else:
        console.print("project already has .loco config")


@app.command()
def clone(
    url: Annotated[str, typer.Argument(help="Git URL or local repo to clone.")],
    dest: Annotated[
        Path | None,
        typer.Argument(help="Directory to create. Defaults to ./<repo-name>."),
    ] = None,
) -> None:
    """Clone a git repo into the workspace volume and write .loco scaffolding."""
    _validate_env()
    from agent_loco.runtime.workspaces import clone_workspace

    if dest is None:
        parent = Path.cwd()
        name = None
    else:
        target = dest.expanduser()
        parent = target.parent
        name = target.name
        if not parent.exists():
            console.print(f"parent does not exist: {parent}")
            raise typer.Exit(code=1)
    try:
        cloned = clone_workspace(url, parent, name=name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"cloned {cloned['path']}")


def _serve_ui(
    workspace: Path,
    settings: Settings,
    *,
    host: str,
    port: int,
    max_concurrent: int,
    default_goal: str | None = None,
    default_create_pr: bool | None = None,
) -> None:
    from agent_loco.web_ui import serve

    console.print(f"Web UI on http://{host}:{port}  (Ctrl+C to stop)")
    if max_concurrent > 1:
        console.print(
            "[yellow]Multiple concurrent tasks will contend for CPU, RAM, "
            "and the local model server.[/yellow]"
        )
    serve(
        workspace,
        settings,
        host=host,
        port=port,
        max_concurrent=max_concurrent,
        default_goal=default_goal,
        default_create_pr=default_create_pr,
    )


@app.command()
def run(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", exists=True, file_okay=False, resolve_path=True),
    ] = Path("."),
    goal: Annotated[str | None, typer.Option("--goal", "-g")] = None,
    model_name: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model on the connected LLM server."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="OpenAI-compatible LLM server (host:port or /v1 URL).",
        ),
    ] = None,
    create_pr: Annotated[bool | None, typer.Option("--create-pr/--no-create-pr")] = None,
    auto_commit: Annotated[bool | None, typer.Option("--commit/--no-commit")] = None,
    web_ui: Annotated[
        bool,
        typer.Option("--web-ui", help="Start the web UI instead of running one cycle."),
    ] = False,
) -> None:
    """Run one improve → test → commit cycle against a project."""
    _validate_env(require=not web_ui)
    settings = _settings(
        model_name=model_name,
        model_base_url=base_url,
        create_pr=create_pr,
        auto_commit=auto_commit,
    )
    if web_ui:
        _serve_ui(
            workspace,
            settings,
            host="127.0.0.1",
            port=8080,
            max_concurrent=1,
            default_goal=goal,
            default_create_pr=create_pr,
        )
        return
    result = run_cycle(
        workspace,
        settings,
        _llm(settings),
        goal,
        cli_create_pr=create_pr,
    )
    console.print(
        f"[bold]{result.status}[/bold] committed={result.committed} "
        f"published={result.published} tests={result.tests_passed}"
    )
    if result.goal:
        console.print(f"goal: {result.goal.splitlines()[0]}")
    if result.summary:
        console.print(result.summary)
    if result.reason and result.status != "success":
        console.print(f"[yellow]{result.reason}[/yellow]")
    elif result.reason and result.reason.startswith("no changes needed"):
        console.print(f"[green]{result.reason}[/green]")
    if result.status in {"failed", "error"}:
        raise typer.Exit(code=2)


@app.command("watch")
def watch_command(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", exists=True, file_okay=False, resolve_path=True),
    ] = Path("."),
    goal: Annotated[str | None, typer.Option("--goal", "-g")] = None,
    interval: Annotated[
        int | None,
        typer.Option("--interval", help="Seconds between cycles."),
    ] = None,
    model_name: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model on the connected LLM server."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="OpenAI-compatible LLM server (host:port or /v1 URL).",
        ),
    ] = None,
) -> None:
    """Keep improving a project on an interval."""
    settings = _settings(
        model_name=model_name,
        model_base_url=base_url,
        watch_interval_seconds=interval,
    )
    try:
        watch_loop(workspace, settings, _llm(settings), goal)
    except KeyboardInterrupt:
        console.print("stopped")


@app.command("ui")
def ui_command(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", exists=True, file_okay=False, resolve_path=True),
    ] = Path("."),
    goal: Annotated[str | None, typer.Option("--goal", "-g")] = None,
    model_name: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model on the connected LLM server."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="OpenAI-compatible LLM server (host:port or /v1 URL).",
        ),
    ] = None,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8080,
    max_concurrent: Annotated[
        int,
        typer.Option(
            "--max-concurrent",
            min=1,
            help="How many cycles may run at once. More than 1 is hard on a local machine.",
        ),
    ] = 1,
    create_pr: Annotated[bool | None, typer.Option("--create-pr/--no-create-pr")] = None,
    auto_commit: Annotated[bool | None, typer.Option("--commit/--no-commit")] = None,
) -> None:
    """Start a local web UI to queue and run tasks.
    
    The UI does not require environment variables set; it will load them from
    a .env, config.yaml, or config.yml file if present, but does not fail if
    they are missing. Users can configure settings via the UI.
    """
    settings = _settings(
        model_name=model_name,
        model_base_url=base_url,
        create_pr=create_pr,
        auto_commit=auto_commit,
    )
    try:
        _serve_ui(
            workspace,
            settings,
            host=host,
            port=port,
            max_concurrent=max_concurrent,
            default_goal=goal,
            default_create_pr=create_pr,
        )
    except KeyboardInterrupt:
        console.print("stopped")
