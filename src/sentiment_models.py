"""Sentiment estimation utilities shared across preprocessing pipelines."""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:  # pragma: no cover
    from textblob import TextBlob  # type: ignore[assignment]
except ImportError:  # pragma: no cover
    TextBlob = None  # type: ignore[misc,assignment]

try:  # pragma: no cover
    from sentida import Sentida  # type: ignore[assignment]
except ImportError:  # pragma: no cover
    Sentida = None  # type: ignore[misc,assignment]

try:  # pragma: no cover
    from nosenti import NoSenti  # type: ignore[assignment]
except ImportError:  # pragma: no cover
    NoSenti = None  # type: ignore[misc,assignment]

try:  # pragma: no cover
    from afinn import Afinn  # type: ignore[assignment]
except ImportError:  # pragma: no cover
    Afinn = None  # type: ignore[misc,assignment]

try:  # pragma: no cover
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[misc,assignment]
    AutoModelForSequenceClassification = None  # type: ignore[misc,assignment]
    AutoTokenizer = None  # type: ignore[misc,assignment]


def textblob_sentiment(text: str) -> float:
    """English sentiment via TextBlob polarity."""
    if not text:
        return 0.0
    if TextBlob is None:
        warnings.warn("TextBlob is not installed; returning neutral sentiment.", RuntimeWarning)
        return 0.0
    try:
        return float(TextBlob(text).sentiment.polarity)  # type: ignore[union-attr]
    except Exception:
        return 0.0


_sentida_analyzer: Optional["Sentida"] = None
_sentida_ready = False


def sentida_sentiment(text: str) -> float:
    """Danish sentiment via SENTIDA lexicon."""
    if not text:
        return 0.0
    global _sentida_analyzer, _sentida_ready
    if Sentida is None:
        warnings.warn("SENTIDA is not installed; returning neutral sentiment.", RuntimeWarning)
        return 0.0
    if not _sentida_ready:
        try:
            _sentida_analyzer = Sentida()
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Failed to initialise SENTIDA ({exc}); returning neutral sentiment.", RuntimeWarning)
            _sentida_analyzer = None
        _sentida_ready = True
    if _sentida_analyzer is None:
        return 0.0
    try:
        return float(_sentida_analyzer.polarity(text))
    except Exception:
        return 0.0


_nosenti_analyzer = None
_nosenti_ready = False


def nosenti_sentiment(text: str) -> float:
    """Norwegian sentiment via NoSenti (falls back to Afinn)."""
    if not text:
        return 0.0
    global _nosenti_analyzer, _nosenti_ready
    if not _nosenti_ready:
        if NoSenti is not None:
            try:
                _nosenti_analyzer = NoSenti()
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"Failed to initialise NoSenti ({exc}); attempting Afinn fallback.", RuntimeWarning)
                _nosenti_analyzer = None
        if _nosenti_analyzer is None and Afinn is not None:
            afinn_languages = ["no", "nb", "nn", "en"]
            for lang in afinn_languages:
                try:
                    _nosenti_analyzer = Afinn(language=lang, emoticons=True)
                    break
                except Exception as exc:  # pragma: no cover
                    warnings.warn(
                        f"Failed to initialise Afinn fallback for language '{lang}' ({exc}).",
                        RuntimeWarning,
                    )
                    _nosenti_analyzer = None
        _nosenti_ready = True

    if _nosenti_analyzer is None:
        raise RuntimeError(
            "NoSenti lexicon sentiment requires either the 'nosenti' or 'afinn' package. "
            "Install one of them and rerun preprocessing."
        )

    try:
        if NoSenti is not None and isinstance(_nosenti_analyzer, NoSenti):  # type: ignore[arg-type]
            polarity = _nosenti_analyzer.polarity(text)
            return float(polarity)
        if Afinn is not None and isinstance(_nosenti_analyzer, Afinn):
            score = _nosenti_analyzer.score(text)
            return float(math.tanh(score / 5.0))
    except Exception:
        return 0.0
    return 0.0


_sentiwordnet_cache: Dict[Path, Dict[str, float]] = {}
_word_pattern = re.compile(r"[^\W\d_]+", re.UNICODE)
DEFAULT_NORWEGIAN_SENTIWORDNET = Path("data/adressa/Norwegian_sentiwordnet.txt")
DEFAULT_DANISH_SENTIWORDNET = Path("data/ebnerd/Danish_sentiwordnet.txt")


def _load_sentiwordnet(path: Path) -> Dict[str, float]:
    resolved = path.expanduser().resolve()
    if resolved in _sentiwordnet_cache:
        return _sentiwordnet_cache[resolved]

    if not resolved.exists():
        raise FileNotFoundError(f"SentiWordNet lexicon file not found: {resolved}")

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                pos_score = float(parts[3])
                neg_score = float(parts[4])
            except ValueError:
                continue
            terms = parts[5].strip().split()
            if not terms:
                continue
            score = pos_score - neg_score
            if score == 0.0:
                continue
            for term in terms:
                term = term.strip()
                if not term:
                    continue
                lemma = term.split("#", 1)[0].strip().lower()
                if not lemma:
                    continue
                sums[lemma] = sums.get(lemma, 0.0) + score
                counts[lemma] = counts.get(lemma, 0) + 1

    lexicon: Dict[str, float] = {}
    for lemma, total in sums.items():
        count = counts.get(lemma, 0)
        if count > 0:
            lexicon[lemma] = total / count

    _sentiwordnet_cache[resolved] = lexicon
    return lexicon


def _sentiwordnet_sentiment(text: str, path: Path) -> float:
    lexicon = _load_sentiwordnet(path)
    if not lexicon:
        return 0.0
    tokens = _word_pattern.findall(text.lower())
    if not tokens:
        return 0.0
    total = 0.0
    hits = 0
    for token in tokens:
        if not token:
            continue
        score = lexicon.get(token)
        if score is None:
            continue
        total += score
        hits += 1
    if hits == 0:
        return 0.0
    return float(total / hits)


def norwegian_swn_sentiment(text: str, *, lexicon_path: Optional[Path] = None) -> float:
    """Sentiment from Norwegian SentiWordNet; averages lemma polarities across the text."""
    if not text:
        return 0.0
    path = lexicon_path or DEFAULT_NORWEGIAN_SENTIWORDNET
    return _sentiwordnet_sentiment(text, path)


def danish_swn_sentiment(text: str, *, lexicon_path: Optional[Path] = None) -> float:
    """Sentiment from Danish SentiWordNet; averages lemma polarities across the text."""
    if not text:
        return 0.0
    path = lexicon_path or DEFAULT_DANISH_SENTIWORDNET
    return _sentiwordnet_sentiment(text, path)


class TransformerSentimentScorer:
    """Multilingual transformer sentiment scorer using CardiffNLP XLM-R model."""

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        *,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.available = False
        self._tokenizer = None
        self._model = None
        self._device = None
        self._positive_idx: Optional[int] = None
        self._negative_idx: Optional[int] = None

        if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
            raise RuntimeError(
                "transformers and torch must be installed to compute transformer sentiment."
            )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
            if torch.cuda.is_available():  # type: ignore[call-arg]
                self._device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
            self._model.to(self._device)
            label_map = {label.lower(): idx for idx, label in self._model.config.id2label.items()}  # type: ignore[union-attr]
            self._positive_idx = self._match_label(label_map, "positive")
            self._negative_idx = self._match_label(label_map, "negative")
            if self._positive_idx is None or self._negative_idx is None:
                raise RuntimeError(
                    "Transformer sentiment model missing positive/negative labels."
                )
            self.available = True
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to load transformer sentiment model '{model_name}': {exc}"
            ) from exc

    @staticmethod
    def _match_label(label_map: dict, key: str) -> Optional[int]:
        for label, idx in label_map.items():
            if key in label:
                return idx
        for label, idx in label_map.items():
            if label.startswith("label_"):
                # Default mapping for models with sequential labels
                if key == "negative" and idx == 0:
                    return idx
                if key == "positive" and idx == 2:
                    return idx
        return None

    def score(self, text: str) -> float:
        """Score a single text in [-1, 1]."""
        if not text:
            return 0.0
        return self.score_many([text])[0]

    def score_many(self, texts: Sequence[str]) -> List[float]:
        """Score multiple texts efficiently."""
        if not self.available or self._tokenizer is None or self._model is None or torch is None:
            raise RuntimeError("Transformer sentiment scorer is not initialised.")

        results: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if not batch:
                continue
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
            for row in probs:
                pos = float(row[self._positive_idx]) if self._positive_idx is not None else 0.0
                neg = float(row[self._negative_idx]) if self._negative_idx is not None else 0.0
                score = max(min(pos - neg, 1.0), -1.0)
                results.append(score)
        return results


__all__ = [
    "textblob_sentiment",
    "sentida_sentiment",
    "nosenti_sentiment",
    "norwegian_swn_sentiment",
    "danish_swn_sentiment",
    "TransformerSentimentScorer",
]
