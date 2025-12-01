from typing import Optional, List
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SimpleSettings(BaseSettings):
    """Simple settings without validators for troubleshooting"""

    # ----- App -----
    app_name: str = "STL-PINN Processor"
    debug: bool = False
    environment: str = "development"

    # ----- API -----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # ----- Database / Cache -----
    database_url: str = "postgresql://jayrk:postgres@localhost:5432/stl_pinn"
    redis_url: str = "redis://localhost:6379/0"
    redis_password: str = "12345"

    # ----- File processing -----
    max_file_size_mb: int = 500
    supported_formats_str: str = "stl,ply,obj,step,stp"
    temp_dir: str = "/tmp/stl_processing"
    output_dir: str = "./outputs"

    # ----- PINN -----
    pinn_model_path: str = "./models/pinn_checkpoint.pt"
    pinn_batch_size: int = 1024
    pinn_epochs: int = 1000
    pinn_learning_rate: float = 0.001
    physics_loss_weight: float = 1.0
    boundary_loss_weight: float = 10.0

    # ----- LLM -----
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ----- Monitoring -----
    enable_metrics: bool = True
    sentry_dsn: Optional[str] = None
    wandb_project: str = "stl-pinn-processing"

    # ----- FreeCAD paths (adjust for your system) -----
    freecad_lib_path: str = "/Applications/FreeCAD.app/Contents/Resources/lib"
    freecad_python_path: str = "/Applications/FreeCAD.app/Contents/Resources/lib/python3.11/site-packages"
    freecad_bin_path: str = "/Applications/FreeCAD.app/Contents/Resources/bin"

    # ----- Security -----
    secret_key: str = "your-secret-key-here-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # ----- Settings config (v2) -----
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # prevents "extra_forbidden" if someone passes unknown keys
    )

    @computed_field
    def supported_formats(self) -> List[str]:
        """Convert supported_formats_str to a normalized list."""
        return [fmt.strip().lower() for fmt in self.supported_formats_str.split(",") if fmt.strip()]

    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.temp_dir, self.output_dir]:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {e}")


# Use simple settings
settings = SimpleSettings()
# Optionally ensure directories exist:
# settings.create_directories()