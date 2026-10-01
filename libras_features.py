"""Transformacao das features de landmarks usada dentro do modelo salvo.

O coletor, o importador, o reconhecedor e o app continuam gerando o vetor bruto
de 126 valores (2 maos x 21 pontos x xyz, mao direita depois esquerda). Este
modulo entra como primeiro passo do Pipeline do scikit-learn, entao qualquer
programa que carregar o .pkl aplica a mesma transformacao sem mudar nada.

Cada mao passa a ser descrita em relacao ao proprio punho e ao tamanho da mao.
Assim o sinal e reconhecido igual perto ou longe da camera, no centro ou no
canto da imagem. Tambem sao adicionadas a presenca de cada mao e a posicao
relativa entre as duas maos, porque isso se perde ao centralizar no punho.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 3
NUM_HANDS = 2
RAW_FEATURES = FEATURES_PER_HAND * NUM_HANDS  # 126


def _split_hands(X: np.ndarray):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    right = X[:, :FEATURES_PER_HAND].reshape(len(X), LANDMARKS_PER_HAND, 3)
    left = X[:, FEATURES_PER_HAND:RAW_FEATURES].reshape(len(X), LANDMARKS_PER_HAND, 3)
    return right, left


def _present(hand: np.ndarray) -> np.ndarray:
    """Mao ausente e gravada como zeros em todas as coordenadas."""
    return np.any(np.abs(hand) > 1e-9, axis=(1, 2))


def _normalize_hand(hand: np.ndarray, present: np.ndarray) -> np.ndarray:
    wrist = hand[:, :1, :]
    rel = hand - wrist
    scale = np.linalg.norm(rel[:, :, :2], axis=2).max(axis=1)
    scale = np.where(scale < 1e-6, 1.0, scale)[:, None, None]
    rel = rel / scale
    rel[~present] = 0.0
    return rel.reshape(len(hand), FEATURES_PER_HAND)


def mirror_raw(X: np.ndarray) -> np.ndarray:
    """Espelha horizontalmente o vetor bruto e troca as maos (aumento de dados)."""
    right, left = _split_hands(X)
    right = right.copy()
    left = left.copy()
    pr, pl = _present(right), _present(left)
    right[pr, :, 0] = 1.0 - right[pr, :, 0]
    left[pl, :, 0] = 1.0 - left[pl, :, 0]
    return np.concatenate(
        [left.reshape(len(left), FEATURES_PER_HAND), right.reshape(len(right), FEATURES_PER_HAND)],
        axis=1,
    )


class HandLandmarkNormalizer(BaseEstimator, TransformerMixin):
    """126 valores brutos -> 126 relativos + 2 presencas + 3 de posicao entre maos."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        right, left = _split_hands(X)
        pr, pl = _present(right), _present(left)
        both = pr & pl
        scale = np.ones(len(right), dtype=np.float32)
        if both.any():
            span_r = np.linalg.norm((right[both] - right[both, :1])[:, :, :2], axis=2).max(axis=1)
            span_l = np.linalg.norm((left[both] - left[both, :1])[:, :, :2], axis=2).max(axis=1)
            scale[both] = np.maximum(np.maximum(span_r, span_l), 1e-6)
        between = np.zeros((len(right), 3), dtype=np.float32)
        between[both] = (left[both, 0] - right[both, 0]) / scale[both, None]
        out = np.concatenate(
            [
                _normalize_hand(right, pr),
                _normalize_hand(left, pl),
                pr[:, None].astype(np.float32),
                pl[:, None].astype(np.float32),
                between,
            ],
            axis=1,
        )
        return out.astype(np.float32)
