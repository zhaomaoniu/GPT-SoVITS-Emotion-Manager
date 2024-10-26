import json
import asyncio
import argparse
from pathlib import Path

from src.gpt_sovits_emotion_manager import Tagger
from src.gpt_sovits_emotion_manager.config import load_config
from src.gpt_sovits_emotion_manager.utils import dump_dataclass
from src.gpt_sovits_emotion_manager.log import setup_logger, log


async def main(file_path: Path):
    config = load_config()

    setup_logger(config)

    tagger = Tagger(config)

    list_annotations = tagger.from_list_file(file_path)

    annotations = await tagger.tag(list_annotations)

    log(
        "INFO",
        f"Tagging finished!",
    )

    if config.tagger.check_duration:
        log(
            "INFO",
            "Checking audio durations. Audios not within the range of 3-10 seconds will be removed.",
        )
        try:
            annotations = tagger.check_duration(annotations)
        except Exception as e:
            log("ERROR", f"Failed to check audio durations", e)

    Path("outputs/emotions").mkdir(parents=True, exist_ok=True)

    output_path = (
        Path("outputs") / "emotions" / f"{file_path.stem}_emotion_annotation.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dump_dataclass(annotations), f, ensure_ascii=False, indent=4)

    log(
        "INFO",
        f"Emotion annotations saved to <c><underline>{output_path.resolve().as_uri()}</underline></c>, click to open.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the emotion tagger.")
    parser.add_argument(
        "--file-path", "-f", type=str, help="The path to the list file."
    )
    args = parser.parse_args()

    if args.file_path is None:
        print(
            "Please specify the path to the list file, e.g. `python run_tagger.py -f /path/to/list_file.txt`"
        )
        exit(1)
    asyncio.run(main(Path(args.file_path)))
