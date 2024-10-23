import yaml
from typing import Literal
from dataclasses import dataclass


@dataclass
class TaggerConfig:
    model: Literal["gemini-1.5-flash", "gemini-1.5-pro"]
    api_key: str
    proxy: str


@dataclass
class Config:
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    tagger: TaggerConfig


def load_config() -> Config:
    def recursive_load(data, cls):
        if hasattr(cls, '__dataclass_fields__'):
            fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
            return cls(**{k: recursive_load(v, fieldtypes[k]) for k, v in data.items()})
        return data

    with open("config.yaml", "r") as f:
        data = yaml.safe_load(f)
        return recursive_load(data, Config)
