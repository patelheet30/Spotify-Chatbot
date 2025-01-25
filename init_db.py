import logging

from app.database.database import init_db
from app.database.models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init() -> None:
    logger.info("Creating initial database")
    init_db()
    logger.info("Database created")


def main() -> None:
    logger.info("Starting database initialization")
    init()
    logger.info("Database initialization complete")


if __name__ == "__main__":
    main()
