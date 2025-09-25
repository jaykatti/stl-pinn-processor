import asyncio
import argparse
from pathlib import Path
import sys

from services.stl_processor import STLProcessor
from services.pinn_service import PINNService
from services.llm_service import LLMService
from services.validation_service import ValidationService
from core.base_processor import ProcessingType
from config.settings import settings
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class STLPINNProcessor:
    """Main application class"""
    
    def __init__(self):
        self.stl_processor = STLProcessor()
        self.pinn_service = PINNService()
        self.llm_service = LLMService()
        self.validation_service = ValidationService()
    
    async def process_file(self, input_path: str, operation: str = "repair",
                          output_format: str = "stl", run_pinn: bool = False):
        """Process a single file"""
        
        logger.info("Starting file processing", 
                   input_path=input_path, operation=operation)
        
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Step 1: STL Processing
        stl_job = self.stl_processor.create_job(
            job_type=ProcessingType.STL_REPAIR,
            input_file_path=input_path,
            parameters={
                "operation_type": operation,
                "output_format": output_format
            }
        )
        
        processed_job = await self.stl_processor.process(stl_job)
        
        if processed_job.status.value == "failed":
            logger.error("STL processing failed", error=processed_job.error_message)
            return processed_job
        
        logger.info("STL processing completed", output_path=processed_job.output_file_path)
        
        # Step 2: Validation
        if processed_job.metadata and "repair_report" in processed_job.metadata:
            repair_report = processed_job.metadata["repair_report"]
            validation_result = await self.validation_service.validate_mesh_quality(
                repair_report.repaired_stats
            )
            
            logger.info("Mesh validation completed",
                       is_valid=validation_result.is_valid,
                       quality_score=validation_result.quality_score)
        
        # Step 3: PINN Analysis (if requested)
        if run_pinn and processed_job.output_file_path:
            logger.info("Starting PINN analysis")
            
            pinn_job = self.pinn_service.create_job(
                job_type=ProcessingType.PINN_ANALYSIS,
                input_file_path=processed_job.output_file_path,
                parameters={
                    "physics_type": "structural",
                    "boundary_conditions": "fixed_bottom",
                    "physics": {
                        "youngs_modulus": 200e9,
                        "poisson_ratio": 0.3,
                        "density": 7850
                    }
                }
            )
            
            pinn_result = await self.pinn_service.process(pinn_job)
            
            if pinn_result.status.value == "completed":
                logger.info("PINN analysis completed", 
                           max_stress=pinn_result.metadata["analysis_results"].max_stress)
            else:
                logger.error("PINN analysis failed", error=pinn_result.error_message)
        
        return processed_job

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="STL-PINN Processor")
    parser.add_argument("input_file", help="Input STL file path")
    parser.add_argument("--operation", default="repair", 
                       choices=["repair", "smooth", "fill_holes", "optimize"],
                       help="Processing operation")
    parser.add_argument("--output-format", default="stl",
                       choices=["stl", "step", "ply", "obj"],
                       help="Output format")
    parser.add_argument("--run-pinn", action="store_true",
                       help="Run PINN analysis after processing")
    parser.add_argument("--api", action="store_true",
                       help="Start API server instead of CLI processing")
    
    args = parser.parse_args()
    
    if args.api:
        # Start API server
        logger.info("Starting API server", host=settings.api_host, port=settings.api_port)
        import uvicorn
        from api.main import app
        
        uvicorn.run(
            app,
            host=settings.api_host,
            port=settings.api_port,
            workers=settings.api_workers
        )
    else:
        # CLI processing
        processor = STLPINNProcessor()
        
        try:
            result = await processor.process_file(
                input_path=args.input_file,
                operation=args.operation,
                output_format=args.output_format,
                run_pinn=args.run_pinn
            )
            
            if result.status.value == "completed":
                print(f"Processing completed successfully!")
                print(f"Output file: {result.output_file_path}")
            else:
                print(f"Processing failed: {result.error_message}")
                sys.exit(1)
                
        except Exception as e:
            logger.error("Processing failed", error=str(e))
            print(f"Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())