from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from pathlib import Path
import shutil
from typing import Optional

from config.settings import settings
from services.stl_processor import STLProcessor
from services.pinn_service import PINNService
from services.validation_service import ValidationService
from core.base_processor import ProcessingType

app = FastAPI(
    title="STL-PINN Processor API",
    description="Production-grade STL processing with PINN analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
stl_processor = STLProcessor()
pinn_service = PINNService()
validation_service = ValidationService()

@app.post("/api/v1/process/stl")
async def process_stl_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_format: str = "stl",
    operation_type: str = "repair"
):
    """Process STL file with intelligent mesh repair"""
    
    # Validate file
    if not file.filename.lower().endswith(('.stl', '.ply', '.obj')):
        raise HTTPException(400, "Unsupported file format")
    
    # Save uploaded file
    temp_path = Path(settings.temp_dir) / file.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create processing job
    job = stl_processor.create_job(
        job_type=ProcessingType.STL_REPAIR,
        input_file_path=str(temp_path),
        parameters={
            "output_format": output_format,
            "operation_type": operation_type
        }
    )
    
    # Start processing in background
    background_tasks.add_task(stl_processor.process, job)
    
    return {"job_id": job.job_id, "status": job.status.value}

@app.post("/api/v1/analyze/pinn")
async def analyze_with_pinn(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    physics_type: str = "structural",
    boundary_conditions: str = "fixed_bottom",
    material: str = "steel"
):
    """Perform PINN-based physics analysis"""
    
    # Save uploaded file
    temp_path = Path(settings.temp_dir) / file.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Material property mapping
    material_properties = {
        "steel": {"youngs_modulus": 200e9, "poisson_ratio": 0.3, "density": 7850},
        "aluminum": {"youngs_modulus": 70e9, "poisson_ratio": 0.33, "density": 2700},
        "titanium": {"youngs_modulus": 110e9, "poisson_ratio": 0.34, "density": 4500}
    }
    
    # Create PINN job
    job = pinn_service.create_job(
        job_type=ProcessingType.PINN_ANALYSIS,
        input_file_path=str(temp_path),
        parameters={
            "physics_type": physics_type,
            "boundary_conditions": boundary_conditions,
            "physics": material_properties.get(material, material_properties["steel"])
        }
    )
    
    # Start analysis in background
    background_tasks.add_task(pinn_service.process, job)
    
    return {"job_id": job.job_id, "status": job.status.value}

@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and results"""
    
    # Check all processors for the job
    for processor in [stl_processor, pinn_service]:
        if job_id in processor.jobs:
            job = processor.jobs[job_id]
            response = {
                "job_id": job.job_id,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat()
            }
            
            if job.error_message:
                response["error"] = job.error_message
            
            if job.output_file_path:
                response["output_file"] = job.output_file_path
            
            if job.metadata:
                response["metadata"] = job.metadata
            
            return response
    
    raise HTTPException(404, "Job not found")

@app.get("/api/v1/download/{job_id}")
async def download_result(job_id: str):
    """Download processed file"""
    
    # Find job and return file
    for processor in [stl_processor, pinn_service]:
        if job_id in processor.jobs:
            job = processor.jobs[job_id]
            if job.output_file_path and Path(job.output_file_path).exists():
                return FileResponse(
                    path=job.output_file_path,
                    filename=Path(job.output_file_path).name,
                    media_type='application/octet-stream'
                )
    
    raise HTTPException(404, "File not found")

@app.post("/api/v1/validate")
async def validate_mesh(file: UploadFile = File(...)):
    """Validate mesh quality"""
    
    # Save and process file for validation
    temp_path = Path(settings.temp_dir) / file.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Quick analysis for validation
    mesh = await stl_processor._load_mesh(str(temp_path))
    stats = stl_processor._analyze_mesh(mesh)
    
    # Validate
    validation_result = await validation_service.validate_mesh_quality(stats)
    
    return {
        "is_valid": validation_result.is_valid,
        "confidence_score": validation_result.confidence_score,
        "quality_score": validation_result.quality_score,
        "issues": validation_result.issues_found,
        "recommendations": validation_result.recommendations
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug
    )