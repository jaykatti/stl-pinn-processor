import openai
import anthropic
from typing import Dict, Any, List, Optional
import json
import asyncio
from enum import Enum

from core.exceptions import LLMServiceError
from models.mesh_models import MeshStatistics

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAMA = "llama"

class LLMService:
    """LLM service for intelligent mesh analysis and code generation"""
    
    def __init__(self):
        self.provider = self._determine_provider()
        self._setup_client()
    
    def _determine_provider(self) -> LLMProvider:
        """Determine which LLM provider to use based on available API keys"""
        if settings.openai_api_key:
            return LLMProvider.OPENAI
        elif settings.anthropic_api_key:
            return LLMProvider.ANTHROPIC
        else:
            return LLMProvider.LLAMA  # Local model
    
    def _setup_client(self):
        """Setup LLM client based on provider"""
        if self.provider == LLMProvider.OPENAI:
            self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        elif self.provider == LLMProvider.ANTHROPIC:
            self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        elif self.provider == LLMProvider.LLAMA:
            # Setup local Llama model (would need llama-cpp-python)
            pass
    
    async def analyze_mesh_quality(self, mesh_stats: MeshStatistics) -> Dict[str, Any]:
        """Analyze mesh quality and recommend operations"""
        
        system_prompt = """
        You are an expert in 3D mesh processing and computational geometry. 
        Analyze the provided mesh statistics and recommend appropriate operations
        to improve mesh quality for engineering analysis and 3D printing.
        
        Consider:
        - Mesh topology (watertight, orientable)
        - Quality metrics (aspect ratio, edge length distribution)
        - Geometric properties (volume, surface area)
        
        Return a JSON response with:
        {
            "quality_assessment": "poor|fair|good|excellent",
            "issues_identified": ["list", "of", "issues"],
            "recommended_operations": [
                {"type": "operation_name", "parameters": {}, "priority": 1-10}
            ],
            "reasoning": "explanation of recommendations"
        }
        """
        
        user_prompt = f"""
        Analyze this mesh:
        - Vertices: {mesh_stats.vertex_count}
        - Triangles: {mesh_stats.triangle_count}
        - Watertight: {mesh_stats.is_watertight}
        - Orientable: {mesh_stats.is_orientable}
        - Surface area: {mesh_stats.surface_area:.6f}
        - Volume: {mesh_stats.volume}
        - Quality metrics: {mesh_stats.quality_metrics}
        """
        
        try:
            response = await self._call_llm(system_prompt, user_prompt)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error("LLM mesh analysis failed", error=str(e))
            return self._fallback_analysis(mesh_stats)
    
    async def evaluate_operation_result(self, operation: Dict[str, Any], 
                                      result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the result of a mesh operation"""
        
        system_prompt = """
        You are evaluating the result of a 3D mesh processing operation.
        Determine if the operation was successful and if processing should continue.
        
        Return JSON:
        {
            "operation_successful": true|false,
            "should_continue": true|false,
            "quality_improvement": "significant|moderate|minimal|none|degraded",
            "next_recommendation": "continue|stop|modify_parameters",
            "reasoning": "explanation"
        }
        """
        
        user_prompt = f"""
        Operation: {operation}
        Result: {result}
        
        Evaluate this operation result.
        """
        
        try:
            response = await self._call_llm(system_prompt, user_prompt)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error("LLM operation evaluation failed", error=str(e))
            return {"operation_successful": True, "should_continue": True}
    
    async def generate_pinn_code(self, requirements: Dict[str, Any]) -> str:
        """Generate PINN code based on physics requirements"""
        
        system_prompt = """
        You are an expert in Physics-Informed Neural Networks (PINNs) and PyTorch.
        Generate optimized PINN code for the given physics problem.
        
        Consider:
        - Appropriate network architecture
        - Physics loss functions
        - Boundary conditions
        - Numerical stability
        
        Return only the Python code, well-commented and production-ready.
        """
        
        user_prompt = f"""
        Generate PINN code for:
        Physics: {requirements.get('physics_type', 'structural_mechanics')}
        Boundary conditions: {requirements.get('boundary_conditions', {})}
        Material properties: {requirements.get('material_properties', {})}
        Geometry constraints: {requirements.get('geometry', {})}
        """
        
        try:
            response = await self._call_llm(system_prompt, user_prompt)
            return response
        except Exception as e:
            logger.error("PINN code generation failed", error=str(e))
            raise LLMServiceError(f"Failed to generate PINN code: {e}")
    
    async def optimize_processing_pipeline(self, mesh_stats: MeshStatistics,
                                         processing_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize processing pipeline based on history and results"""
        
        system_prompt = """
        You are optimizing a 3D mesh processing pipeline.
        Based on the mesh characteristics and processing history, 
        suggest pipeline optimizations.
        
        Return JSON:
        {
            "pipeline_efficiency": "low|medium|high",
            "bottlenecks": ["list", "of", "bottlenecks"],
            "optimizations": [
                {"type": "optimization_type", "description": "details", "impact": "low|medium|high"}
            ],
            "parameter_suggestions": {}
        }
        """
        
        user_prompt = f"""
        Mesh: {mesh_stats}
        Processing history: {processing_history}
        
        Optimize the processing pipeline.
        """
        
        try:
            response = await self._call_llm(system_prompt, user_prompt)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error("Pipeline optimization failed", error=str(e))
            return {"pipeline_efficiency": "medium", "optimizations": []}
    
    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM API based on provider"""
        
        if self.provider == LLMProvider.OPENAI:
            response = await self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
            return response.choices[0].message.content
            
        elif self.provider == LLMProvider.ANTHROPIC:
            response = await self.client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
            
        elif self.provider == LLMProvider.LLAMA:
            # Local Llama implementation would go here
            raise NotImplementedError("Local Llama not implemented yet")
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning("Failed to parse LLM JSON response", error=str(e), response=response)
            return {}
    
    def _fallback_analysis(self, mesh_stats: MeshStatistics) -> Dict[str, Any]:
        """Fallback analysis when LLM fails"""
        operations = []
        
        # Basic heuristic-based recommendations
        if not mesh_stats.is_watertight:
            operations.append({"type": "fill_holes", "parameters": {}, "priority": 9})
        
        if mesh_stats.quality_metrics.get("mean_aspect_ratio", 1.0) > 5.0:
            operations.append({"type": "smooth_mesh", "parameters": {"iterations": 5}, "priority": 7})
        
        if mesh_stats.vertex_count > 100000:
            operations.append({"type": "decimate_mesh", "parameters": {"target_ratio": 0.7}, "priority": 5})
        
        operations.append({"type": "remove_duplicates", "parameters": {}, "priority": 8})
        operations.append({"type": "repair_normals", "parameters": {}, "priority": 6})
        
        return {
            "quality_assessment": "fair",
            "issues_identified": ["topology_issues", "quality_metrics"],
            "recommended_operations": operations,
            "reasoning": "Fallback heuristic-based analysis"
        }