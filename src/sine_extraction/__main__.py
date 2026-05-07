"""CLI entry point — uv run python -m sine_extraction <subcommand>."""

from sine_extraction.cli import cli

__all__ = ["cli"]

if __name__ == "__main__":
    cli()
