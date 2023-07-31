import os
import logging

from GraphTranslation.config.config import Config


def setup_logging():
    config = Config()
    os.makedirs(config.logging_folder, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | [%(levelname)s] | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.logging_folder, "relationBaseKGQA.log"), encoding="utf8"),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)