"""Runtime configuration loaded from environment variables and .env files."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from core.exceptions import ConfigurationError

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings used by the GUI and translation pipeline."""

    api_key: str | None
    base_url: str | None
    model_name: str
    input_dir: Path
    output_dir: Path
    default_source_lang: str
    default_target_lang: str
    default_ocr_lang_code: str
    default_max_tokens: int
    pdf_renderer_path: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables with project-local defaults."""
        project_root = Path(__file__).resolve().parents[1]
        logger.debug("Loading settings from environment. project_root=%s", project_root)
        return cls(
            api_key=os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL"),
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            input_dir=Path(os.getenv("INPUT_DIR", project_root / "input")),
            output_dir=Path(os.getenv("OUTPUT_DIR", project_root / "output")),
            default_source_lang=os.getenv("SOURCE_LANG", "English"),
            default_target_lang=os.getenv("TARGET_LANG", "فارسی"),
            default_ocr_lang_code=os.getenv("OCR_LANG_CODE", "eng"),
            default_max_tokens=int(os.getenv("MAX_TOKENS", "2500")),
            pdf_renderer_path=os.getenv("PDF_RENDERER_PATH"),
        )

    def require_api_key(self) -> str:
        """Return the configured API key or raise a configuration error."""
        if not self.api_key:
            raise ConfigurationError("API_KEY or OPENAI_API_KEY is not set in the environment.")
        logger.debug("API key configuration is present")
        return self.api_key

    def ensure_directories(self) -> None:
        """Create configured input and output directories if they do not exist."""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured project directories exist. input_dir=%s output_dir=%s", self.input_dir, self.output_dir)


settings = Settings.from_env()
