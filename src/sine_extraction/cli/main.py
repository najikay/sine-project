"""Top-level Click group and subcommand registration."""

from __future__ import annotations

import click

from sine_extraction.cli.commands import cmd_generate, cmd_train
from sine_extraction.cli.eval_vis import cmd_evaluate, cmd_visualize
from sine_extraction.constants import DEFAULT_CONFIG_PATH

_DATA_DIR_OPT = click.option(
    "--data-dir",
    default="artifacts/data",
    show_default=True,
    help="Directory for X.npy / y.npy",
)


@click.group()
@click.option(
    "--config",
    default=str(DEFAULT_CONFIG_PATH),
    show_default=True,
    help="Path to config.yaml",
)
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """Sine Wave Extraction — data generation, training, evaluation, visualization."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@_DATA_DIR_OPT
@click.pass_context
def all(ctx: click.Context, data_dir: str) -> None:
    """Run the full pipeline: generate → train → evaluate → visualize."""
    ctx.invoke(cmd_generate, data_dir=data_dir)
    ctx.invoke(cmd_train, data_dir=data_dir)
    ctx.invoke(cmd_evaluate, data_dir=data_dir)
    ctx.invoke(cmd_visualize, data_dir=data_dir)


cli.add_command(cmd_generate, name="generate")
cli.add_command(cmd_train, name="train")
cli.add_command(cmd_evaluate, name="evaluate")
cli.add_command(cmd_visualize, name="visualize")
