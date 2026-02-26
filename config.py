from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class AppConfig(BaseSettings):
    """
    Centralized configuration for ESM2 fastapi app.
    Using BaseSettings to manage app-specific environment.

    In Kubernetes deployment manifest, using <envFrom: - configMapRef> allows
    import of these settings from a Kubernetes ConfigMap

    amp_mode:   Automatic Mixed Precision (AMP) is a new feature for Blackwell's to autodetect
                areas where lower-precision tensors can be handled
                for high-speed 8-bit or 16-bit inference for VRAM optimization.

    return_residue_embeddings: Swap to true if the residue-level embeddings are needed.
    """
    mock_gpu: bool = Field(default=False, description="Enable CPU mock mode for local testing")
    mock_gpu_count: int = Field(default=4, description="Number of GPUs to simulate in mock mode")
    workers_per_gpu: int = Field(default=1, description="Number of worker processes to spawn per GPU")
    log_level: str = Field(default="INFO", description="Logging verbosity level")
    inference_precision: str = Field(default="float16", description="Precision for model inference (float16, bfloat16, float32)")
    output_precision: str = Field(default="float32", description="Precision for output embeddings (float32, float16)")

    model_name: str = Field(default="facebook/esm2_t33_650M_UR50D", description="HuggingFace model ID")
    max_sequence_length: int = Field(default=1022, description="Max context window for ESM-2")
    return_residue_embeddings: bool = Field(default=False, description="Return per-residue embeddings if True")
    amp_mode: str = Field(default="none", description="AMP precision (none, float16, bfloat16, fp8)")

    #Inject/superimpose settings that were defined in a Kubernetes ConfigMap and imported to the env.
    model_config = SettingsConfigDict(env_prefix="ESM2_", env_file=".env", extra="ignore")


settings = AppConfig()
