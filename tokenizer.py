# tokenizer.py
import re
from collections import defaultdict
import json
import os
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer"""
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
    })
    max_vocab_size: int = 10000
    min_freq: int = 2
    cache_size: int = 10000
    lowercase: bool = True
    strip_accents: bool = True
    vocab_file: str = "tokenizer_vocab.json"


class EnhancedTokenizer:
    """Enhanced tokenizer with improved features and error handling"""

    def __init__(self, config):
        """Initialize tokenizer with configuration"""
        self.config = config
        self.model_config = config.model
        self.tokenizer_config = TokenizerConfig()

        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.word_frequencies: Dict[str, int] = defaultdict(int)
        self.is_trained: bool = False

        # Initialize special tokens
        self._initialize_special_tokens()

        # Load vocabulary if exists
        if os.path.exists(self.tokenizer_config.vocab_file):
            self.load_vocab()

    def _initialize_special_tokens(self) -> None:
        """Initialize special tokens in vocabulary"""
        for token, idx in self.tokenizer_config.special_tokens.items():
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token

    @lru_cache(maxsize=10000)
    def _clean_text(self, text: str) -> str:
        """Clean text with caching"""
        if not isinstance(text, str):
            text = str(text)

        if self.tokenizer_config.lowercase:
            text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Handle contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'m", " am", text)

        return text

    @lru_cache(maxsize=10000)
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with caching"""
        try:
            # Clean the text
            text = self._clean_text(text)

            # Add spaces around punctuation but keep it
            text = re.sub(r'([.,!?(){}[\]"\'=+\-*/^<>])', r' \1 ', text)

            # Handle math operators and numbers
            text = re.sub(r'(\d+)', r' \1 ', text)

            # Split and filter empty strings
            tokens = [token for token in text.split() if token]

            return tokens
        except Exception as e:
            logger.error(f"Error in tokenization: {str(e)}")
            return []

    def build_vocab_from_df(
            self,
            df: pd.DataFrame,
            text_column: str,
            min_freq: Optional[int] = None,
            max_vocab_size: Optional[int] = None
    ) -> None:
        """Build vocabulary from DataFrame"""
        if self.is_trained and os.path.exists(self.tokenizer_config.vocab_file):
            logger.info("Loading existing vocabulary...")
            self.load_vocab()
            return

        logger.info("Building vocabulary from DataFrame...")
        min_freq = min_freq or self.tokenizer_config.min_freq
        max_vocab_size = max_vocab_size or self.tokenizer_config.max_vocab_size

        # Count word frequencies
        for text in tqdm(df[text_column], desc="Counting word frequencies"):
            tokens = self.tokenize(str(text))
            for token in tokens:
                self.word_frequencies[token] += 1

        # Sort words by frequency
        sorted_words = sorted(
            self.word_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Add words that meet criteria to vocab
        current_idx = len(self.tokenizer_config.special_tokens)
        for word, freq in sorted_words:
            if freq < min_freq:
                break
            if current_idx >= max_vocab_size:
                break
            if word not in self.vocab:
                self.vocab[word] = current_idx
                self.reverse_vocab[current_idx] = word
                current_idx += 1

        self.is_trained = True
        self.save_vocab()
        logger.info(f"Vocabulary built with {len(self.vocab)} tokens")

    def encode(
            self,
            text: str,
            max_length: Optional[int] = None
    ) -> List[int]:
        """Convert text to token ids"""
        if not self.is_trained:
            raise ValueError("Tokenizer needs to be trained first!")

        max_length = max_length or self.model_config.max_seq_len

        try:
            tokens = self.tokenize(text)

            # Start with BOS token
            token_ids = [self.tokenizer_config.special_tokens["<BOS>"]]

            # Add token ids
            token_ids.extend(
                self.vocab.get(token, self.tokenizer_config.special_tokens["<UNK>"])
                for token in tokens
            )

            # Add EOS token
            token_ids.append(self.tokenizer_config.special_tokens["<EOS>"])

            # Handle length
            if len(token_ids) < max_length:
                # Pad sequence
                token_ids.extend([self.tokenizer_config.special_tokens["<PAD>"]] *
                                 (max_length - len(token_ids)))
            else:
                # Truncate sequence
                token_ids = token_ids[:max_length - 1] + [self.tokenizer_config.special_tokens["<EOS>"]]

            return token_ids
        except Exception as e:
            logger.error(f"Error in encoding: {str(e)}")
            return [self.tokenizer_config.special_tokens["<UNK>"]] * max_length

    def encode_batch(
            self,
            texts: List[str],
            max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Encode a batch of texts"""
        encodings = [self.encode(text, max_length) for text in tqdm(texts, desc="Encoding texts")]
        return torch.tensor(encodings)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token ids back to text"""
        try:
            tokens = []
            for id in token_ids:
                token = self.reverse_vocab.get(id, "<UNK>")
                if skip_special_tokens and token in self.tokenizer_config.special_tokens:
                    continue
                tokens.append(token)

            # Handle spacing around punctuation
            text = " ".join(tokens)
            text = re.sub(r'\s+([.,!?(){}[\]"\'=+\-*/^<>])', r'\1', text)
            return text
        except Exception as e:
            logger.error(f"Error in decoding: {str(e)}")
            return ""

    def save_vocab(self, path: Optional[str] = None) -> None:
        """Save vocabulary to file"""
        path = path or self.tokenizer_config.vocab_file
        try:
            save_dict = {
                'vocab': self.vocab,
                'reverse_vocab': {str(k): v for k, v in self.reverse_vocab.items()},
                'word_frequencies': dict(self.word_frequencies),
                'is_trained': self.is_trained,
            }

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(save_dict, f, ensure_ascii=False, indent=2)

            logger.info(f"Vocabulary saved to {path}")
        except Exception as e:
            logger.error(f"Error saving vocabulary: {str(e)}")

    def load_vocab(self, path: Optional[str] = None) -> bool:
        """Load vocabulary from file"""
        path = path or self.tokenizer_config.vocab_file

        try:
            if not os.path.exists(path):
                logger.warning(f"No vocabulary file found at {path}")
                return False

            with open(path, 'r', encoding='utf-8') as f:
                save_dict = json.load(f)

            self.vocab = save_dict['vocab']
            self.reverse_vocab = {int(k): v for k, v in save_dict['reverse_vocab'].items()}
            self.word_frequencies = defaultdict(int, save_dict['word_frequencies'])
            self.is_trained = save_dict['is_trained']

            logger.info(f"Vocabulary loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading vocabulary: {str(e)}")
            return False


def prepare_sentiment_data(
        config,
        sample_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, EnhancedTokenizer]:
    """Prepare sentiment data from configuration"""
    try:
        # Read data
        df = pd.read_csv(config.data.data_path)
        df = df[['review', 'sentiment']]

        if sample_size:
            df = df.sample(n=sample_size, random_state=config.data.random_seed)
        elif config.data.sample_size:
            df = df.sample(n=config.data.sample_size, random_state=config.data.random_seed)

        # Map sentiments
        sentiment_dict = {
            'positive': 1,
            'negative': 0
        }
        df['sentiment'] = df['sentiment'].map(sentiment_dict)

        # Initialize and train tokenizer
        tokenizer = EnhancedTokenizer(config)
        tokenizer.build_vocab_from_df(
            df,
            'review',
            min_freq=config.data.min_freq
        )

        # Encode all texts
        X = tokenizer.encode_batch(df['review'].tolist())
        Y = torch.tensor(df['sentiment'].values)

        logger.info(f"Prepared dataset with {len(df)} samples")
        logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")

        return X, Y, tokenizer
    except Exception as e:
        logger.error(f"Error preparing sentiment data: {str(e)}")
        raise


if __name__ == "__main__":
    from config import Config

    # Example usage
    config = Config()
    X, Y, tokenizer = prepare_sentiment_data(config)

    # Test tokenization
    sample_text = "This is a great movie!"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print(f"\nSample text: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")