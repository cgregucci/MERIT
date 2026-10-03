"""Standalone, torch-free CQA benchmark generator for ULTRA.

Implements the reduction-tier strategy of the +H benchmarks from their published
definitions, for both transductive and inductive datasets, emitting queries/answers
in ULTRA's on-disk format.

Depends only on the Python standard library. Does NOT import torch or
any `ultra.*` module (those pull in torch); `STRUCT2TYPE` is re-declared locally.
"""
