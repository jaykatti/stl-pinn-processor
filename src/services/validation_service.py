from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    confidence_score: float
    issues_found: List[str]
    recommendations: List[str]
    quality_score: float
    
class ValidationService:
    """Service for validating mesh quality and PINN results"""
    
    def __init__(self):
        self.llm_service = LLMService()
    
    async def validate_mesh_quality(self, mesh_stats: MeshStatistics, 
                                  requirements: Dict[str, Any] = None) -> ValidationResult:
        """Validate mesh quality against requirements"""
        
        issues = []
        recommendations = []
        quality_score = 100.0
        
        # Check topology
        if not mesh_stats.is_watertight:
            issues.append("Mesh is not watertight")
            recommendations.append("Fill holes and repair mesh topology")
            quality_score -= 20
        
        if not mesh_stats.is_orientable:
            issues.append("Mesh has orientation issues")
            recommendations.append("Fix triangle orientation")
            quality_score -= 15
        
        # Check quality metrics
        mean_aspect_ratio = mesh_stats.quality_metrics.get("mean_aspect_ratio", 1.0)
        if mean_aspect_ratio > 10.0:
            issues.append(f"High aspect ratio triangles (avg: {mean_aspect_ratio:.2f})")
            recommendations.append("Smooth mesh or remesh problematic areas")
            quality_score -= 15
        
        # Check geometric properties
        if mesh_stats.volume is None or mesh_stats.volume <= 0:
            issues.append("Invalid or zero volume")
            recommendations.append("Repair mesh to create valid solid")
            quality_score -= 25
        
        # Check requirements compliance
        if requirements:
            req_issues, req_recommendations = await self._check_requirements_compliance(
                mesh_stats, requirements
            )
            issues.extend(req_issues)
            recommendations.extend(req_recommendations)
        
        # Get LLM validation
        llm_validation = await self._get_llm_validation(mesh_stats, issues)
        
        confidence_score = max(0.0, min(1.0, quality_score / 100.0))
        is_valid = len(issues) == 0 or confidence_score > 0.7
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            issues_found=issues,
            recommendations=recommendations,
            quality_score=quality_score
        )
    
    async def validate_pinn_results(self, analysis_result: PINNAnalysisResult,
                                   mesh_stats: MeshStatistics) -> ValidationResult:
        """Validate PINN analysis results"""
        
        issues = []
        recommendations = []
        quality_score = 100.0
        
        # Check for numerical issues
        if np.any(np.isnan(analysis_result.displacement_field)):
            issues.append("NaN values in displacement field")
            quality_score -= 30
        
        if np.any(np.isinf(analysis_result.stress_field)):
            issues.append("Infinite values in stress field")
            quality_score -= 30
        
        # Check physical reasonableness
        if analysis_result.max_displacement > 1.0:  # 1 meter seems unreasonable for most cases
            issues.append("Unreasonably large displacements")
            recommendations.append("Check boundary conditions and material properties")
            quality_score -= 20
        
        if analysis_result.safety_factor < 1.0:
            issues.append("Safety factor below 1.0 - potential failure")
            recommendations.append("Increase material strength or reduce loading")
            quality_score -= 25
        elif analysis_result.safety_factor < 2.0:
            issues.append("Low safety factor")
            recommendations.append("Consider design modifications for higher safety margin")
            quality_score -= 10
        
        # Check stress concentrations
        stress_std = np.std(analysis_result.von_mises_stress)
        stress_mean = np.mean(analysis_result.von_mises_stress)
        if stress_std > 2.0 * stress_mean:
            issues.append("High stress concentrations detected")
            recommendations.append("Add fillets or modify geometry to reduce stress concentrations")
            quality_score -= 15
        
        confidence_score = max(0.0, min(1.0, quality_score / 100.0))
        is_valid = len(issues) == 0 or confidence_score > 0.6
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            issues_found=issues,
            recommendations=recommendations,
            quality_score=quality_score
        )
    
    async def _check_requirements_compliance(self, mesh_stats: MeshStatistics,
                                           requirements: Dict[str, Any]) -> tuple:
        """Check if mesh meets specified requirements"""
        issues = []
        recommendations = []
        
        # Check size requirements
        if "max_vertices" in requirements:
            if mesh_stats.vertex_count > requirements["max_vertices"]:
                issues.append(f"Too many vertices: {mesh_stats.vertex_count} > {requirements['max_vertices']}")
                recommendations.append("Decimate mesh to reduce vertex count")
        
        if "min_volume" in requirements:
            if mesh_stats.volume and mesh_stats.volume < requirements["min_volume"]:
                issues.append(f"Volume too small: {mesh_stats.volume} < {requirements['min_volume']}")
        
        # Check quality requirements
        if "max_aspect_ratio" in requirements:
            mean_ar = mesh_stats.quality_metrics.get("mean_aspect_ratio", 1.0)
            if mean_ar > requirements["max_aspect_ratio"]:
                issues.append(f"Aspect ratio too high: {mean_ar} > {requirements['max_aspect_ratio']}")
                recommendations.append("Improve mesh quality through smoothing or remeshing")
        
        return issues, recommendations
    
    async def _get_llm_validation(self, mesh_stats: MeshStatistics, 
                                 issues: List[str]) -> Dict[str, Any]:
        """Get LLM-based validation insights"""
        try:
            system_prompt = """
            You are validating 3D mesh quality for engineering analysis.
            Provide additional insights and validation based on the mesh statistics and identified issues.
            
            Return JSON:
            {
                "additional_issues": ["list", "of", "additional", "issues"],
                "severity_assessment": "critical|moderate|minor",
                "validation_confidence": 0.0-1.0,
                "expert_recommendations": ["list", "of", "recommendations"]
            }
            """
            
            user_prompt = f"""
            Mesh Statistics: {mesh_stats}
            Identified Issues: {issues}
            
            Provide expert validation assessment.
            """
            
            response = await self.llm_service._call_llm(system_prompt, user_prompt)
            return self.llm_service._parse_json_response(response)
            
        except Exception as e:
            logger.warning("LLM validation failed", error=str(e))
            return {"additional_issues": [], "validation_confidence": 0.8}