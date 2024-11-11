# config.py
from dataclasses import dataclass
from typing import Optional
import torch
from pathlib import Path


@dataclass
class ModelConfig:
    d_model: int = 512
    num_head: int = 8
    drop_prob: float = 0.3
    ffn_hidden: int = 512
    num_layers: int = 2
    vocab_size: int = 10000
    num_classes: int = 2
    max_seq_len: int = 100


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 10
    clip_value: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2


@dataclass
class DataConfig:
    data_path: Path = Path("IMDB Dataset.csv")
    train_size: float = 0.8
    random_seed: int = 42
    min_freq: int = 2
    sample_size: Optional[int] = 5000  # Set to None to use full dataset


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()

    def __post_init__(self):
        # Convert data_path to Path object if it's a string
        if isinstance(self.data.data_path, str):
            self.data.data_path = Path(self.data.data_path)

    @property
    def device(self) -> torch.device:
        return torch.device(self.training.device)

    def save(self, path: str = "config.json"):
        """Save configuration to a JSON file"""
        import json

        # Convert the config to a dictionary
        config_dict = {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": {k: str(v) if isinstance(v, Path) else v
                     for k, v in self.data.__dict__.items()}
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str = "config.json") -> 'Config':
        """Load configuration from a JSON file"""
        import json

        with open(path) as f:
            config_dict = json.load(f)

        model_config = ModelConfig(**config_dict["model"])
        training_config = TrainingConfig(**config_dict["training"])
        data_config = DataConfig(**config_dict["data"])

        return cls(
            model=model_config,
            training=training_config,
            data=data_config
        )

    def display(self):
        """Display the current configuration"""
        from pprint import pprint
        print("Current Configuration:")
        print("\nModel Config:")
        pprint(self.model.__dict__)
        print("\nTraining Config:")
        pprint(self.training.__dict__)
        print("\nData Config:")
        pprint(self.data.__dict__)