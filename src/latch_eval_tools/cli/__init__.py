"""Command line interface for latch-eval-tools.

The implementation lives in sibling modules: ``app`` (argument parser + the
``main`` entry point), ``run`` (the ``run`` subcommand), ``harnesses`` (harness
dispatch), and ``preflight`` (checks). The ``latch-eval`` console script points
directly at ``latch_eval_tools.cli.app:main``.
"""
