import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class VisualizationUtils:
    """Utilities for visualizing meshes and analysis results"""
    
    @staticmethod
    def create_mesh_quality_report(mesh_stats: Dict[str, Any], 
                                 quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Create comprehensive mesh quality visualization"""
        
        # Create plots
        plots = {}
        
        # Quality metrics bar chart
        metric_names = list(quality_metrics.keys())
        metric_values = list(quality_metrics.values())
        
        fig_metrics = px.bar(
            x=metric_names, 
            y=metric_values,
            title="Mesh Quality Metrics",
            labels={"x": "Metric", "y": "Value"}
        )
        plots["quality_metrics"] = fig_metrics
        
        # Aspect ratio distribution (if available)
        if "aspect_ratios" in mesh_stats:
            fig_aspect = px.histogram(
                x=mesh_stats["aspect_ratios"],
                title="Aspect Ratio Distribution",
                labels={"x": "Aspect Ratio", "y": "Count"}
            )
            plots["aspect_ratio_distribution"] = fig_aspect
        
        return plots
    
    @staticmethod
    def create_pinn_analysis_plots(analysis_result) -> Dict[str, Any]:
        """Create PINN analysis visualization plots"""
        
        plots = {}
        
        # Displacement magnitude plot
        displacement_magnitude = np.linalg.norm(analysis_result.displacement_field, axis=1)
        
        fig_displacement = go.Figure()
        fig_displacement.add_trace(go.Scatter3d(
            x=analysis_result.displacement_field[:, 0],
            y=analysis_result.displacement_field[:, 1], 
            z=analysis_result.displacement_field[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=displacement_magnitude,
                colorscale='Viridis',
                colorbar=dict(title="Displacement Magnitude")
            ),
            name='Displacement Field'
        ))
        fig_displacement.update_layout(title="Displacement Field Visualization")
        plots["displacement_field"] = fig_displacement
        
        # Von Mises stress plot
        fig_stress = go.Figure()
        fig_stress.add_trace(go.Histogram(
            x=analysis_result.von_mises_stress,
            nbinsx=50,
            name="Von Mises Stress Distribution"
        ))
        fig_stress.update_layout(
            title="Von Mises Stress Distribution",
            xaxis_title="Stress (Pa)",
            yaxis_title="Frequency"
        )
        plots["stress_distribution"] = fig_stress
        
        # Convergence history
        if analysis_result.convergence_history:
            fig_convergence = go.Figure()
            fig_convergence.add_trace(go.Scatter(
                y=analysis_result.convergence_history,
                mode='lines',
                name="Training Loss"
            ))
            fig_convergence.update_layout(
                title="PINN Training Convergence",
                xaxis_title="Epoch",
                yaxis_title="Loss"
            )
            plots["convergence"] = fig_convergence
        
        return plots
    
    @staticmethod
    def create_comparison_plots(original_stats: Dict[str, Any],
                               processed_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Create before/after comparison plots"""
        
        plots = {}
        
        # Metrics comparison
        metrics = ["vertex_count", "triangle_count", "surface_area"]
        original_values = [original_stats.get(m, 0) for m in metrics]
        processed_values = [processed_stats.get(m, 0) for m in metrics]
        
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            name='Original',
            x=metrics,
            y=original_values
        ))
        fig_comparison.add_trace(go.Bar(
            name='Processed',
            x=metrics,
            y=processed_values
        ))
        fig_comparison.update_layout(
            title="Mesh Processing Comparison",
            barmode='group'
        )
        plots["metrics_comparison"] = fig_comparison
        
        return plots