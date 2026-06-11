"""Command line interface for latch-eval-tools.

The implementation is split across ``app`` (parser + entry point), ``run`` (the
``run`` subcommand), ``harnesses`` (harness dispatch), and ``preflight`` (checks).
``main`` is re-exported here so the ``latch-eval`` console script entry point
(``latch_eval_tools.cli:main``) keeps working.
"""

from latch_eval_tools.cli.app import main

__all__ = ["main"]
