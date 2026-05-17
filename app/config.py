from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Service root = parent of the `app/` package; resolves the same whether the
# service runs from a source checkout or from the Docker image's /opt/lp/service.
SERVICE_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_root: Path = Field(default=SERVICE_ROOT)
    yolov5_root: Path = Field(default=Path("/opt/lp/yolov5"))
    lp_detector_weights: Path = Field(
        default=SERVICE_ROOT / "model" / "LP_detector_nano_61.pt"
    )
    lp_ocr_weights: Path = Field(
        default=SERVICE_ROOT / "model" / "LP_ocr_nano_62.pt"
    )

    ocr_conf: float = 0.60
    detector_imgsz: int = 640
    device: str = "cuda"

    max_batch: int = 8
    max_image_bytes: int = 10_000_000


settings = Settings()
