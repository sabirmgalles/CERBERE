"""
url_vocab.py — Vocabulaire et encodage caractère-niveau pour les URLs.

Séparé de train_url_lstm.py à dessein : ce module n'a AUCUNE dépendance
vers torch. Cela permet à api.py et à la suite de tests de fonctionner
même sur une machine sans PyTorch installé (mode heuristique pur) — seul
train_url_lstm.py (l'entraînement) et le chargement effectif du modèle
exigent torch, et ces deux points l'importent en local, à la demande.
"""
from __future__ import annotations

MAX_LEN = 200
VOCAB = list(" abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%")
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}  # 0 = padding/inconnu


def encode_url(url: str, max_len: int = MAX_LEN) -> list[int]:
    """Encode une URL en une séquence fixe d'entiers (troncature/padding à max_len)."""
    url = (url or "").lower()[:max_len]
    ids = [CHAR2IDX.get(c, 0) for c in url]
    ids += [0] * (max_len - len(ids))
    return ids
