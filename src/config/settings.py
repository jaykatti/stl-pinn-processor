from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    # Application settings
    app_name: str = "STL-PINN Processor"
    debug: bool = False
    environment: str = "development"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Database settings
    database_url: str = "postgresql://user:pass@localhost:5432/stl_pinn"
    redis_url: str = "redis://localhost:6379/0"
    
    # File processing settings
    max_file_size_mb: int = 500
    supported_formats: List[str] = ["stl", "ply", "obj", "step", "stp"]
    temp_dir: str = "/tmp/stl_processing"
    output_dir: str = "./outputs"
    
    # PINN settings
    pinn_model_path: str = "./models/pinn_checkpoint.pt"
    pinn_batch_size: int = 1024
    pinn_epochs: int = 1000
    pinn_learning_rate: float = 0.001
    physics_loss_weight: float = 1.0
    boundary_loss_weight: float = 10.0
    
    # LLM settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_model: str = "gpt-4"  # or "claude-3-sonnet", "llama-2-70b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    
    # Monitoring settings
    enable_metrics: bool = True
    sentry_dsn: Optional[str] = None
    wandb_project: str = "stl-pinn-processing"
    
    # FreeCAD settings
    freecad_lib_path: str = "/Applications/FreeCAD.app/Contents/Resources/lib"
    freecad_python_path: str = "/Applications/FreeCAD.app/Contents/Resources/lib/python3.11/site-packages"
    freecad_bin_path: str = "/Applications/FreeCAD.app/Contents/Resources/bin"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()