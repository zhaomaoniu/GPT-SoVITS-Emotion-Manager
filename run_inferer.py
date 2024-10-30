import json
import time
import asyncio
import argparse
from pathlib import Path
from httpx import TimeoutException

from src.gpt_sovits_emotion_manager import Inferer
from src.gpt_sovits_emotion_manager.config import load_config
from src.gpt_sovits_emotion_manager.utils import emotion_to_str
from src.gpt_sovits_emotion_manager.log import setup_logger, log
from src.gpt_sovits_emotion_manager.models import EmotionAnnotation, Emotion


def log_prompt(emotion_types: str):
    log(
        "INFO",
        "Emotions should be in the format of `<y>type</y>:<magenta>intensity</magenta>`.",
    )
    log(
        "INFO",
        f"Valid types: <y>{emotion_types}</y>",
    )
    log(
        "INFO",
        "Valid intensities: <magenta>low, moderate, high</magenta>",
    )
    log(
        "INFO",
        "Example: `<y>joy</y>:<magenta>low</magenta>,<y>fear</y>:<magenta>moderate</magenta>`",
    )


async def main(file_path: Path):
    config = load_config()

    setup_logger(config)

    emotion_annotations = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            emotions = [
                Emotion(type=emotion["type"], intensity=emotion["intensity"])
                for emotion in item["emotions"]
            ]
            emotion_annotations.append(
                EmotionAnnotation(
                    emotions=emotions,
                    text=item["text"],
                    file=item["file"],
                    language=item["language"],
                )
            )

    if len(emotion_annotations) == 0:
        log("ERROR", "No emotion annotations found in the file.")
        return None

    inferer = Inferer(emotion_annotations, config)

    while True:
        text = input("Text: ")
        while True:
            emotions_text = input("Emotions (leave empty for LLM): ").strip().lower()
            if not emotions_text:
                emotions = None
                log("INFO", "No emotions specified, using LLM to infer emotions.")
                try:
                    emotions = await inferer.get_emotion_from_text(text)
                    log("INFO", f"Emotions inferred by LLM: {emotion_to_str(emotions)}")
                except Exception as e:
                    log(
                        "WARNING",
                        f"Failed to infer emotions by LLM, using default emotion: {config.emotion_types[0]}:low",
                    )
                    emotions = [Emotion(type=config.emotion_types[0], intensity="low")]
                break
            else:
                try:
                    emotions = [
                        Emotion(
                            type=emotion.split(":")[0], intensity=emotion.split(":")[1]
                        )
                        for emotion in emotions_text.split(",")
                    ]
                except Exception as e:
                    log("ERROR", "Invalid emotions!")
                    log_prompt(", ".join(config.emotion_types))
                    continue
                # check if all emotions are valid
                if not all(
                    emotion.type
                    in config.emotion_types
                    and emotion.intensity in {"low", "moderate", "high"}
                    for emotion in emotions
                ):
                    log("ERROR", "Invalid emotions!")
                    log_prompt(", ".join(config.emotion_types))
                else:
                    break

        while True:
            language = input("Language: ").strip().lower()
            if language not in {"zh", "ja", "en", "ko", "yue"}:
                log("ERROR", "Invalid language!")
                log("INFO", "Supported languages: <y>zh, ja, en, ko, yue</y>")
            else:
                break

        log("INFO", "Generating content...")
        try:
            result = await inferer.generate(text, language, emotions)
        except TimeoutException:
            log("ERROR", "Timeout occurred, please try again.")
            continue
        except Exception as e:
            log("ERROR", "Failed to generate content", e)
            continue
        output_path = (
            Path("outputs")
            / "audios"
            / f"{int(time.time())}_{emotion_to_str(emotions).replace(':', '_')}.{config.inference.media_type}"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(result)

        log(
            "INFO",
            f"Audio saved to <c><underline>{output_path.absolute().as_uri()}</underline></c>, click to open.",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the GPT-SoVITS emotion inference."
    )
    parser.add_argument(
        "--file-path", "-f", type=str, help="The path to the emotion annotations file."
    )
    args = parser.parse_args()

    if args.file_path is None:
        print(
            "Please specify the path to the emotion annotations file, e.g. `python run_inferer.py -f /path/to/emotion_annotations.json`"
        )
        exit(1)

    if not Path(args.file_path).exists():
        print(f"File not found: {args.file_path}")
        exit(1)

    asyncio.run(main(Path(args.file_path)))
