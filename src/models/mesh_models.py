from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class MeshStatistics:
    vertex_count: int
    triangle_count: int
    is_watertight: bool
    is_orientable: bool
    surface_area: float
    volume: Optional[float]
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    quality_metrics: Dict[str, float]

@dataclass
class MeshRepairReport:
    original_stats: MeshStatistics
    repaired_stats: MeshStatistics
    operations_performed: List[str]
    holes_filled: int
    vertices_removed: int
    triangles_removed: int
    success: bool
    processing_time: float

@dataclass
class PhysicsProperties:
    """Physical properties for PINN analysis"""
    density: float = 1.0  # kg/m³
    youngs_modulus: float = 200e9  # Pa (steel default)
    poisson_ratio: float = 0.3
    thermal_conductivity: float = 50.0  # W/m·K
    specific_heat: float = 500.0  # J/kg·K
    thermal_expansion: float = 12e-6  # 1/K