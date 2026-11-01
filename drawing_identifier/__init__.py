"""Agentic identification of hand drawings: shapes, their nesting/connections and the text inside and near them."""

from .agents import AnalysisResult, Orchestrator, analyze
from .config import AppConfig, load_config
from .schema import (
    BBox,
    Connection,
    ConnectionType,
    DiagramGraph,
    Endpoint,
    Relation,
    RelationType,
    Shape,
    ShapeType,
    TextItem,
    TextPlacement,
)

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "Orchestrator",
    "analyze",
    "AppConfig",
    "load_config",
    "BBox",
    "Connection",
    "ConnectionType",
    "DiagramGraph",
    "Endpoint",
    "Relation",
    "RelationType",
    "Shape",
    "ShapeType",
    "TextItem",
    "TextPlacement",
]
