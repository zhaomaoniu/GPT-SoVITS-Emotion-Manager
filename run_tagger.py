import json
import asyncio
from pathlib import Path

from src.gpt_sovits_emotion_manager import Tagger
from src.gpt_sovits_emotion_manager.config import load_config
from src.gpt_sovits_emotion_manager.utils import dump_dataclass
from src.gpt_sovits_emotion_manager.log import setup_logger, logger


async def main():
    config = load_config()

    setup_logger(config)

    tagger = Tagger(config.tagger)

    file_path = Path(input("Enter the path to the list file: ").strip())

    list_annotations = tagger.from_list_file(file_path)

    annotations = await tagger.tag(list_annotations)

    Path("outputs").mkdir(exist_ok=True)

    output_path = Path("outputs") / f"{file_path.stem}_emotion_annotation.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dump_dataclass(annotations), f, ensure_ascii=False, indent=4)

    logger.info(f"Emotion annotations have been written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
