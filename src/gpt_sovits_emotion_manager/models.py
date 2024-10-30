from typing import List, Literal
from dataclasses import dataclass


@dataclass
class Emotion:
    type: str
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
