import numpy as np
from typing import Tuple, List, Dict, Any
import open3d as o3d
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.transform import Rotation

class GeometryUtils:
    """Utility functions for geometric computations"""
    
    @staticmethod
    def compute_mesh_properties(mesh: o3d.geometry.TriangleMesh) -> Dict[str, Any]:
        """Compute comprehensive mesh properties"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        properties = {}
        
        # Basic properties
        properties["vertex_count"] = len(vertices)
        properties["triangle_count"] = len(triangles)
        properties["edge_count"] = len(triangles) * 3 // 2  # Approximation
        
        # Bounding box
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        properties["bounding_box"] = {
            "min": bbox_min.tolist(),
            "max": bbox_max.tolist(),
            "size": (bbox_max - bbox_min).tolist()
        }
        
        # Center of mass
        properties["center_of_mass"] = vertices.mean(axis=0).tolist()
        
        # Surface area and volume
        try:
            properties["surface_area"] = mesh.get_surface_area()
            if mesh.is_watertight():
                properties["volume"] = mesh.get_volume()
            else:
                properties["volume"] = None
        except:
            properties["surface_area"] = 0.0
            properties["volume"] = None
        
        return properties
    
    @staticmethod
    def compute_mesh_quality_metrics(mesh: o3d.geometry.TriangleMesh) -> Dict[str, float]:
        """Compute advanced mesh quality metrics"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        metrics = {}
        
        # Triangle quality metrics
        aspect_ratios = []
        areas = []
        angles = []
        edge_lengths = []
        
        for triangle in triangles:
            v1, v2, v3 = vertices[triangle]
            
            # Edge vectors
            e1 = v2 - v1
            e2 = v3 - v2
            e3 = v1 - v3
            
            # Edge lengths
            l1, l2, l3 = np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(e3)
            edge_lengths.extend([l1, l2, l3])
            
            # Triangle area
            area = 0.5 * np.linalg.norm(np.cross(e1, -e3))
            areas.append(area)
            
            # Aspect ratio
            if min(l1, l2, l3) > 0:
                aspect_ratio = max(l1, l2, l3) / min(l1, l2, l3)
            else:
                aspect_ratio = float('inf')
            aspect_ratios.append(aspect_ratio)
            
            # Triangle angles
            if l1 > 0 and l2 > 0 and l3 > 0:
                # Law of cosines
                angle1 = np.arccos(np.clip((l2**2 + l3**2 - l1**2) / (2*l2*l3), -1, 1))
                angle2 = np.arccos(np.clip((l1**2 + l3**2 - l2**2) / (2*l1*l3), -1, 1))
                angle3 = np.pi - angle1 - angle2
                angles.extend([angle1, angle2, angle3])
        
        # Compute statistics
        metrics["aspect_ratio_mean"] = np.mean(aspect_ratios)
        metrics["aspect_ratio_std"] = np.std(aspect_ratios)
        metrics["aspect_ratio_max"] = np.max(aspect_ratios)
        
        metrics["edge_length_mean"] = np.mean(edge_lengths)
        metrics["edge_length_std"] = np.std(edge_lengths)
        metrics["edge_length_min"] = np.min(edge_lengths)
        metrics["edge_length_max"] = np.max(edge_lengths)
        
        metrics["triangle_area_mean"] = np.mean(areas)
        metrics["triangle_area_std"] = np.std(areas)
        metrics["triangle_area_min"] = np.min(areas)
        
        metrics["angle_min_degrees"] = np.degrees(np.min(angles))
        metrics["angle_max_degrees"] = np.degrees(np.max(angles))
        
        return metrics
    
    @staticmethod
    def detect_features(mesh: o3d.geometry.TriangleMesh, 
                       angle_threshold: float = 30.0) -> Dict[str, Any]:
        """Detect geometric features like edges, corners"""
        
        # Compute normals if not present
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        
        triangle_normals = np.asarray(mesh.triangle_normals)
        triangles = np.asarray(mesh.triangles)
        
        # Build adjacency information
        edge_triangle_map = {}
        
        for i, triangle in enumerate(triangles):
            edges = [(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])]
            for edge in edges:
                edge_key = tuple(sorted(edge))
                if edge_key not in edge_triangle_map:
                    edge_triangle_map[edge_key] = []
                edge_triangle_map[edge_key].append(i)
        
        # Detect sharp edges
        sharp_edges = []
        threshold_cos = np.cos(np.radians(angle_threshold))
        
        for edge, triangle_indices in edge_triangle_map.items():
            if len(triangle_indices) == 2:
                # Interior edge
                normal1 = triangle_normals[triangle_indices[0]]
                normal2 = triangle_normals[triangle_indices[1]]
                
                dot_product = np.dot(normal1, normal2)
                if dot_product < threshold_cos:
                    sharp_edges.append(edge)
        
        return {
            "sharp_edges": sharp_edges,
            "edge_count": len(edge_triangle_map),
            "boundary_edges": len([e for e, t in edge_triangle_map.items() if len(t) == 1]),
            "feature_edge_count": len(sharp_edges)
        }
    
    @staticmethod
    def compute_curvature(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        """Compute mesh curvature at vertices"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Build vertex-triangle adjacency
        vertex_triangles = [[] for _ in range(len(vertices))]
        for i, triangle in enumerate(triangles):
            for vertex_idx in triangle:
                vertex_triangles[vertex_idx].append(i)
        
        curvatures = np.zeros(len(vertices))
        
        for i, vertex in enumerate(vertices):
            adjacent_triangles = vertex_triangles[i]
            
            if len(adjacent_triangles) < 3:
                continue
            
            # Get neighboring vertices
            neighbors = set()
            for tri_idx in adjacent_triangles:
                triangle = triangles[tri_idx]
                for v_idx in triangle:
                    if v_idx != i:
                        neighbors.add(v_idx)
            
            if len(neighbors) < 3:
                continue
            
            # Compute mean curvature using cotangent weights
            neighbor_vertices = vertices[list(neighbors)]
            center = vertices[i]
            
            # Simplified curvature estimation
            vectors_to_neighbors = neighbor_vertices - center
            distances = np.linalg.norm(vectors_to_neighbors, axis=1)
            
            if len(distances) > 0:
                curvatures[i] = np.std(distances) / np.mean(distances)
        
        return curvatures

class MeshOptimizer:
    """Advanced mesh optimization algorithms"""
    
    @staticmethod
    async def optimize_for_simulation(mesh: o3d.geometry.TriangleMesh,
                                    target_quality: float = 0.8) -> o3d.geometry.TriangleMesh:
        """Optimize mesh for finite element analysis"""
        
        # Current quality assessment
        quality_metrics = GeometryUtils.compute_mesh_quality_metrics(mesh)
        current_quality = 1.0 / (1.0 + quality_metrics["aspect_ratio_mean"])
        
        optimized_mesh = mesh
        
        if current_quality < target_quality:
            # Apply optimization strategies
            
            # 1. Laplacian smoothing
            optimized_mesh = optimized_mesh.filter_smooth_laplacian(number_of_iterations=5)
            
            # 2. Remove degenerate triangles
            optimized_mesh.remove_degenerate_triangles()
            optimized_mesh.remove_unreferenced_vertices()
            
            # 3. Taubin smoothing for better preservation
            optimized_mesh = optimized_mesh.filter_smooth_taubin(number_of_iterations=10)
            
            # 4. Check if decimation is needed for very dense meshes
            if len(optimized_mesh.triangles) > 50000:
                target_triangles = min(50000, int(len(optimized_mesh.triangles) * 0.8))
                optimized_mesh = optimized_mesh.simplify_quadric_decimation(target_triangles)
        
        return optimized_mesh
    
    @staticmethod
    def remesh_adaptive(mesh: o3d.geometry.TriangleMesh,
                       curvature_threshold: float = 0.1) -> o3d.geometry.TriangleMesh:
        """Adaptive remeshing based on curvature"""
        
        # Compute curvature
        curvatures = GeometryUtils.compute_curvature(mesh)
        
        # Identify high curvature regions
        high_curvature_mask = curvatures > curvature_threshold
        
        # For now, return smoothed version
        # In production, would implement proper adaptive remeshing
        if np.any(high_curvature_mask):
            return mesh.filter_smooth_laplacian(number_of_iterations=3)
        else:
            return mesh
