import open3d as o3d
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import asyncio
import structlog

from core.base_processor import BaseProcessor, ProcessingJob, ProcessingStatus, ProcessingType
from core.exceptions import MeshProcessingError
from models.mesh_models import MeshStatistics, MeshRepairReport
from services.freecad_service import FreeCADService
from services.llm_service import LLMService

logger = structlog.get_logger(__name__)

class STLProcessor(BaseProcessor):
    """Advanced STL processing with LLM-guided optimization"""
    
    def __init__(self):
        super().__init__("STL Processor")
        self.freecad_service = FreeCADService()
        self.llm_service = LLMService()
    
    async def process(self, job: ProcessingJob) -> ProcessingJob:
        """Process STL file with intelligent optimization"""
        try:
            job.status = ProcessingStatus.RUNNING
            self.update_job_progress(job.job_id, 0.1, ProcessingStatus.RUNNING)
            
            # Load and analyze mesh
            logger.info("Loading STL file", file_path=job.input_file_path)
            mesh = await self._load_mesh(job.input_file_path)
            original_stats = self._analyze_mesh(mesh)
            
            self.update_job_progress(job.job_id, 0.2)
            
            # Get LLM analysis and recommendations
            llm_analysis = await self.llm_service.analyze_mesh_quality(original_stats)
            recommended_operations = llm_analysis.get("recommended_operations", [])
            
            logger.info("LLM recommended operations", operations=recommended_operations)
            
            self.update_job_progress(job.job_id, 0.3)
            
            # Apply recommended repairs
            repaired_mesh = mesh
            operations_performed = []
            
            for operation in recommended_operations:
                operation_result = await self._apply_operation(repaired_mesh, operation)
                if operation_result["success"]:
                    repaired_mesh = operation_result["mesh"]
                    operations_performed.append(operation)
                    
                    # Get LLM evaluation of the operation
                    evaluation = await self.llm_service.evaluate_operation_result(
                        operation, operation_result
                    )
                    
                    if not evaluation.get("should_continue", True):
                        logger.warning("LLM recommends stopping operations", 
                                     operation=operation, evaluation=evaluation)
                        break
            
            self.update_job_progress(job.job_id, 0.8)
            
            # Final validation and export
            final_stats = self._analyze_mesh(repaired_mesh)
            output_path = await self._export_mesh(repaired_mesh, job)
            
            # Create repair report
            repair_report = MeshRepairReport(
                original_stats=original_stats,
                repaired_stats=final_stats,
                operations_performed=operations_performed,
                holes_filled=0,  # Will be populated by specific operations
                vertices_removed=original_stats.vertex_count - final_stats.vertex_count,
                triangles_removed=original_stats.triangle_count - final_stats.triangle_count,
                success=True,
                processing_time=0.0  # Will be calculated
            )
            
            job.output_file_path = output_path
            job.metadata = {"repair_report": repair_report}
            job.status = ProcessingStatus.COMPLETED
            job.progress = 1.0
            
            return job
            
        except Exception as e:
            logger.error("STL processing failed", error=str(e), job_id=job.job_id)
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            return job
    
    async def validate_input(self, input_data: Any) -> bool:
        """Validate STL file input"""
        try:
            if isinstance(input_data, (str, Path)):
                file_path = Path(input_data)
                return file_path.exists() and file_path.suffix.lower() in ['.stl', '.ply', '.obj']
            return False
        except Exception:
            return False
    
    async def _load_mesh(self, file_path: str) -> o3d.geometry.TriangleMesh:
        """Load mesh file asynchronously"""
        def load_sync():
            mesh = o3d.io.read_triangle_mesh(file_path)
            if len(mesh.vertices) == 0:
                raise MeshProcessingError(f"Failed to load mesh from {file_path}")
            return mesh
        
        return await asyncio.get_event_loop().run_in_executor(None, load_sync)
    
    def _analyze_mesh(self, mesh: o3d.geometry.TriangleMesh) -> MeshStatistics:
        """Analyze mesh and compute statistics"""
        # Compute bounding box
        bbox = mesh.get_axis_aligned_bounding_box()
        bbox_min = bbox.get_min_bound()
        bbox_max = bbox.get_max_bound()
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(mesh)
        
        # Compute volume safely
        volume = None
        try:
            if mesh.is_watertight():
                volume = mesh.get_volume()
        except:
            pass
        
        return MeshStatistics(
            vertex_count=len(mesh.vertices),
            triangle_count=len(mesh.triangles),
            is_watertight=mesh.is_watertight(),
            is_orientable=mesh.is_orientable(),
            surface_area=mesh.get_surface_area(),
            volume=volume,
            bounding_box=((bbox_min[0], bbox_min[1], bbox_min[2]), 
                         (bbox_max[0], bbox_max[1], bbox_max[2])),
            quality_metrics=quality_metrics
        )
    
    def _compute_quality_metrics(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, float]:
        """Compute advanced mesh quality metrics"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        quality_metrics = {}
        
        # Aspect ratio analysis
        aspect_ratios = []
        for triangle in triangles:
            v1, v2, v3 = vertices[triangle]
            
            # Calculate edge lengths
            edge1 = np.linalg.norm(v2 - v1)
            edge2 = np.linalg.norm(v3 - v2) 
            edge3 = np.linalg.norm(v1 - v3)
            
            # Calculate aspect ratio (longest edge / shortest edge)
            edges = [edge1, edge2, edge3]
            aspect_ratio = max(edges) / min(edges) if min(edges) > 0 else 0
            aspect_ratios.append(aspect_ratio)
        
        quality_metrics["mean_aspect_ratio"] = np.mean(aspect_ratios)
        quality_metrics["max_aspect_ratio"] = np.max(aspect_ratios)
        quality_metrics["aspect_ratio_std"] = np.std(aspect_ratios)
        
        # Edge length analysis
        edge_lengths = []
        for triangle in triangles:
            v1, v2, v3 = vertices[triangle]
            edge_lengths.extend([
                np.linalg.norm(v2 - v1),
                np.linalg.norm(v3 - v2),
                np.linalg.norm(v1 - v3)
            ])
        
        quality_metrics["mean_edge_length"] = np.mean(edge_lengths)
        quality_metrics["edge_length_std"] = np.std(edge_lengths)
        quality_metrics["min_edge_length"] = np.min(edge_lengths)
        quality_metrics["max_edge_length"] = np.max(edge_lengths)
        
        return quality_metrics
    
    async def _apply_operation(self, mesh: o3d.geometry.TriangleMesh, 
                              operation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a mesh operation based on LLM recommendation"""
        operation_type = operation.get("type")
        parameters = operation.get("parameters", {})
        
        try:
            if operation_type == "remove_duplicates":
                return await self._remove_duplicates(mesh)
            elif operation_type == "fill_holes":
                return await self._fill_holes(mesh, parameters)
            elif operation_type == "smooth_mesh":
                return await self._smooth_mesh(mesh, parameters)
            elif operation_type == "decimate_mesh":
                return await self._decimate_mesh(mesh, parameters)
            elif operation_type == "repair_normals":
                return await self._repair_normals(mesh)
            elif operation_type == "freecad_repair":
                return await self.freecad_service.advanced_repair(mesh, parameters)
            else:
                return {"success": False, "error": f"Unknown operation: {operation_type}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _remove_duplicates(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, Any]:
        """Remove duplicate vertices and triangles"""
        def remove_sync():
            original_vertices = len(mesh.vertices)
            original_triangles = len(mesh.triangles)
            
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
            
            return {
                "success": True,
                "mesh": mesh,
                "vertices_removed": original_vertices - len(mesh.vertices),
                "triangles_removed": original_triangles - len(mesh.triangles)
            }
        
        return await asyncio.get_event_loop().run_in_executor(None, remove_sync)
    
    async def _repair_normals(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, Any]:
        """Repair mesh normals"""
        def repair_sync():
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            try:
                mesh.orient_triangles()
                return {"success": True, "mesh": mesh, "normals_repaired": True}
            except:
                return {"success": True, "mesh": mesh, "normals_repaired": False}
        
        return await asyncio.get_event_loop().run_in_executor(None, repair_sync)
    
    async def _export_mesh(self, mesh: o3d.geometry.TriangleMesh, 
                          job: ProcessingJob) -> str:
        """Export processed mesh"""
        output_format = job.parameters.get("output_format", "stl")
        input_path = Path(job.input_file_path)
        
        output_path = Path(settings.output_dir) / f"{input_path.stem}_processed.{output_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        def export_sync():
            success = o3d.io.write_triangle_mesh(str(output_path), mesh)
            if not success:
                raise MeshProcessingError(f"Failed to export mesh to {output_path}")
            return str(output_path)
        
        return await asyncio.get_event_loop().run_in_executor(None, export_sync)