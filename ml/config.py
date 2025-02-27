import os
import sys

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]


def _format_message(record):
    # process record to generate
    t = record["time"]
    level = record["level"]

    file = record["file"]
    function = record["function"]
    line = record["line"]

    message = record["message"]

    # escape tags in message, tags are "<some_tag>"
    if function.startswith("<") and function.endswith(">"):
        function = rf"\{function}"

    message.replace("<", r"\<")

    # strip the path to the project root and replace it with a dot for brevity in both file and message
    file = str(file.path).replace(str(PROJ_ROOT), "$PROJECT_ROOT")
    message = message.replace(str(PROJ_ROOT), "$PROJECT_ROOT")

    return (
        f"<green>{t:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        f"<level>{level: <8}</level> | "
        f"<cyan>{file}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n"
    )


logger.remove(0)

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.add(lambda msg: tqdm.write(msg, end=""), format=_format_message, colorize=True)
except ModuleNotFoundError:
    logger.add(sys.stderr, format=_format_message, colorize=True)

# Paths
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Data directories
DATA_DIR = os.getenv("DATA_DIR", PROJ_ROOT / "data")
logger.info(f"DATA_DIR path is: {DATA_DIR}")

SAMPLE_DATA_DIR = os.getenv("SAMPLE_DATA_DIR", DATA_DIR / "sample")
logger.info(f"SAMPLE_DATA_DIR path is: {SAMPLE_DATA_DIR}")

CONFIGS_DIR = os.getenv("CONFIGS_DIR", PROJ_ROOT / "configs")
logger.info(f"CONFIGS_DIR path is: {CONFIGS_DIR}")

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = os.getenv("MODELS_DIR", PROJ_ROOT / "models")

# Report directories
REPORTS_DIR = os.getenv("REPORTS_DIR", PROJ_ROOT / "reports")
FIGURES_DIR = REPORTS_DIR / "figures"

LOGS_DIR = os.getenv("LOGS_DIR", PROJ_ROOT / "logs")
logger.info(f"LOGS_DIR path is: {LOGS_DIR}")

# ML Logging and Monitoring
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
if WANDB_ENTITY is None:
    logger.info("WANDB_ENTITY is not set. This will log to the default entity.")
else:
    logger.info(f"WANDB_ENTITY is: {WANDB_ENTITY}")

WANDB_PROJECT = os.getenv("WANDB_PROJECT", None)
if WANDB_PROJECT is None:
    logger.warning(
        f"WANDB_PROJECT is not set. Defaulting to: {Path(__file__).resolve().parents[1].name}"
    )
    WANDB_PROJECT = Path(__file__).resolve().parents[1].name
else:
    logger.info(f"WANDB_PROJECT is: {WANDB_PROJECT}")

if os.getenv("WANDB_MODE") == "disabled":
    logger.info("WANDB_MODE is disabled. Disabling WandB logging.")
    # set WANDB_SILENT to true to disable logging
    os.environ["WANDB_SILENT"] = "true"
    logger.info("WandB is made silent.")
