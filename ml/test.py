from loguru import logger

from ml.scheduling import Schedule

from ml.util import STORE

logger.info(f"STORE has: {STORE}")

logger.info(Schedule.parse("CosSched(0.1, 1e-6)"))
