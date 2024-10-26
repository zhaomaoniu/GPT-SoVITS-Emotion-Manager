from typing import List, Literal
from dataclasses import dataclass


@dataclass
class Emotion:
    type: Literal[
        "joy",  # 喜悦
        "fear",  # 恐惧
        "surprise",  # 惊讶
        "sadness",  # 悲伤
        "disgust",  # 厌恶
        "anger",  # 愤怒
        "neutral",  # 中性
        "confusion",  # 困惑
    ]
    intensity: Literal["low", "moderate", "high"]


@dataclass
class EmotionAnnotation:
    file: str
    text: str
    language: Literal["zh", "ja", "en", "ko", "yue"]
    emotions: List[Emotion]


@dataclass
class ListFileAnnotation:
    path: str
    speaker: str
    language: Literal["zh", "ja", "en", "ko", "yue"]
    text: str
