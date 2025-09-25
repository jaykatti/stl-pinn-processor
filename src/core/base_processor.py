from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime

class ProcessingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingType(Enum):
    STL_REPAIR = "stl_repair"
    MESH_OPTIMIZATION = "mesh_optimization"
    PINN_ANALYSIS = "pinn_analysis"
    PHYSICS_SIMULATION = "physics_simulation"
    QUALITY_VALIDATION = "quality_validation"

@dataclass
class ProcessingJob:
    job_id: str
    job_type: ProcessingType
    status: ProcessingStatus
    created_at: datetime
    updated_at: datetime
    input_file_path: str
    output_file_path: Optional[str] = None
    parameters: Dict[str, Any] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class BaseProcessor(ABC):
    """Base class for all processing services"""
    
    def __init__(self, name: str):
        self.name = name
        self.jobs: Dict[str, ProcessingJob] = {}
    
    @abstractmethod
    async def process(self, job: ProcessingJob) -> ProcessingJob:
        """Process a job and return updated job status"""
        pass
    
    @abstractmethod
    async def validate_input(self, input_data: Any) -> bool:
        """Validate input data before processing"""
        pass
    
    def create_job(self, job_type: ProcessingType, input_file_path: str, 
                   parameters: Dict[str, Any] = None) -> ProcessingJob:
        """Create a new processing job"""
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            job_id=job_id,
            job_type=job_type,
            status=ProcessingStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            input_file_path=input_file_path,
            parameters=parameters or {}
        )
        self.jobs[job_id] = job
        return job
    
    def update_job_progress(self, job_id: str, progress: float, 
                           status: Optional[ProcessingStatus] = None):
        """Update job progress"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.progress = progress
            job.updated_at = datetime.now()
            if status:
                job.status = status