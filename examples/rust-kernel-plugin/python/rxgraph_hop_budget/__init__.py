"""Downstream rxgraph extension exposing the native `hop_budget` kernel."""

from . import _native
from rxgraph.plugin import export_api

export_api(globals(), _native)
