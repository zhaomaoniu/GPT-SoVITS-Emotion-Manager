import yaml
from typing import Literal, List
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    base_url: str
    use_aux_ref: bool
    max_aux_refs: int
    top_k: int
    top_p: float
    temperature: float
    text_split_method: str
    batch_size: int
    batch_threshold: float
    split_bucket: bool
    speed_factor: float
    fragment_interval: float
    streaming_mode: bool
    seed: int
    parallel_infer: bool
    repetition_penalty: float
    media_type: str


@dataclass
class TaggerConfig:
    check_duration: bool


@dataclass
class LLMConfig:
    model: Literal["gemini-1.5-flash", "gemini-1.5-pro"]
    api_key: str
    proxy: str


@dataclass
class Config:
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    emotion_types: List[str]
    inference: InferenceConfig
    tagger: TaggerConfig
    llm: LLMConfig


def load_config() -> Config:
    def recursive_load(data, cls):
        if hasattr(cls, "__dataclass_fields__"):
            fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
            return cls(**{k: recursive_load(v, fieldtypes[k]) for k, v in data.items()})
        return data

    with open("config.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return recursive_load(data, Config)
