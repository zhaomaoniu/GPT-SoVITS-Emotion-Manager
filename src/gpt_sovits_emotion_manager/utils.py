import wave
from typing import Any, List
from dataclasses import asdict, is_dataclass

from .models import Emotion


def dump_dataclass(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: dump_dataclass(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [dump_dataclass(i) for i in obj]
    if isinstance(obj, dict):
        return {k: dump_dataclass(v) for k, v in obj.items()}
    return obj


def get_audio_duration(file_path: str) -> float:
    with wave.open(file_path, "rb") as f:
        return f.getnframes() / f.getframerate()


def equal_emotions(emotions: List[Emotion], emotions_: List[Emotion]) -> bool:
    if len(emotions) != len(emotions_):
        return False
    emotions.sort(key=lambda x: x.type)
    emotions_.sort(key=lambda x: x.type)
    for emotion, emotion_ in zip(emotions, emotions_):
        if emotion.type != emotion_.type or emotion.intensity != emotion_.intensity:
            return False
    return True


def emotion_to_str(emotions: List[Emotion]) -> str:
    return ",".join([f"{emotion.type}:{emotion.intensity}" for emotion in emotions])
