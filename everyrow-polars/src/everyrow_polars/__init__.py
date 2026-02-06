"""Polars integration for everyrow: LLM-powered DataFrame operations.

Import this module to register the `everyrow` namespace on Polars DataFrames:

    import polars as pl
    import everyrow_polars  # registers df.everyrow namespace

    df = pl.read_csv("leads.csv")
    screened = df.everyrow.screen("Remote-friendly senior role with disclosed salary")
"""

from everyrow_polars.namespace import EveryrowNamespace

__all__ = ["EveryrowNamespace"]
