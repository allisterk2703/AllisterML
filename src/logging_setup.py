import inspect
import logging
import sys
from pathlib import Path


class EmojiFormatter(logging.Formatter):
    EMOJIS = {
        logging.DEBUG: "🐛",
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🔥",
    }

    def format(self, record):
        record.emoji = self.EMOJIS.get(record.levelno, "")
        return super().format(record)


def setup_logging(dataset_name: str, level=logging.INFO) -> None:
    log_dir = Path(f"../data/{dataset_name}/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s - %(levelname)s - %(emoji)s %(message)s"
    formatter = EmojiFormatter(fmt)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)

    file = logging.FileHandler(log_dir / "log.log", encoding="utf-8")
    file.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=[stream, file], force=True)

    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Logging initialisé; dossier: {log_dir}")
