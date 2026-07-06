"""Vega regime gate — deliberately dumb market-state filter (STRATEGY.md §5).

Simple, auditable components only: SPY vs 200DMA, VIX bands, universe breadth,
crypto fear/greed. Clever regime models are the most overfit component of
retail systems — this module must stay boring.
"""
