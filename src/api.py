"""
api.py — Backend FastAPI de CERBÈRE.

Expose :
  GET  /health
  POST /analyze/email
  POST /analyze/url

Comportement :
  - Les 3 têtes heuristiques (texte, URL, en-têtes) tournent TOUJOURS —
    l'API reste donc pleinement fonctionnelle sans aucun modèle entraîné,
    et même sans PyTorch/transformers installés du tout (dégradation
    propre : chaque import ML est local à sa fonction et protégé par
    try/except ImportError).
  - Si un modèle DistilBERT existe dans models/text_model/ (produit par
    train_bert.py), sa probabilité est fusionnée avec le score heuristique
    du texte (moyenne pondérée).
  - Si un modèle LSTM existe dans models/url_model/ (produit par
    train_url_lstm.py), même principe pour le score URL.
  - Le HTML des e-mails est parsé avec BeautifulSoup (preprocessing.py)
    pour extraire les liens et détecter les décalages texte affiché / URL réelle.

Lancement :
  uvicorn src.api:app --reload --port 8000
"""
from __future__ import annotations
import os
import json
import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from . import heuristics as H
from . import preprocessing as P
from .url_vocab import encode_url

MAX_BODY_LEN = 20_000     # protège contre les payloads abusifs
MAX_FIELD_LEN = 500

TEXT_MODEL_DIR = os.environ.get("CERBERE_TEXT_MODEL", "models/text_model")
URL_MODEL_DIR = os.environ.get("CERBERE_URL_MODEL", "models/url_model")

app = FastAPI(title="CERBÈRE API", version="1.0")

# En démonstration : ouvert à toutes les origines pour que le frontend
# (ouvert en file:// ou servi ailleurs) puisse appeler l'API facilement.
# En production, restreindre allow_origins à votre propre domaine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- schémas
class EmailRequest(BaseModel):
    display_name: str = ""
    from_addr: str = ""
    reply_to: str = ""
    subject: str = ""
    body: str = ""          # texte brut OU HTML

    @field_validator("display_name", "from_addr", "reply_to", "subject")
    @classmethod
    def _cap_field_len(cls, v: str) -> str:
        return (v or "")[:MAX_FIELD_LEN]

    @field_validator("body")
    @classmethod
    def _cap_body_len(cls, v: str) -> str:
        return (v or "")[:MAX_BODY_LEN]


class UrlRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        v = (v or "").strip()[:MAX_FIELD_LEN]
        if not v:
            raise ValueError("L'URL ne peut pas être vide.")
        if not re.match(r"^(https?://|www\.)?[^\s]+\.[a-zA-Z]{2,}", v):
            raise ValueError("Format d'URL non reconnu.")
        return v


# ---------------------------------------------------------- chargement ML
_text_pipeline = None
_url_model = None
_url_vocab = None


def _try_load_text_model():
    global _text_pipeline
    if not os.path.isdir(TEXT_MODEL_DIR):
        print(f"[CERBÈRE] Aucun modèle texte trouvé dans {TEXT_MODEL_DIR} — mode heuristique seul pour la tête NLP.")
        return
    try:
        from transformers import pipeline
        _text_pipeline = pipeline("text-classification", model=TEXT_MODEL_DIR, tokenizer=TEXT_MODEL_DIR, top_k=None)
        print(f"[CERBÈRE] Modèle texte chargé depuis {TEXT_MODEL_DIR}.")
    except Exception as e:
        print(f"[CERBÈRE] Échec du chargement du modèle texte ({e}) — repli sur l'heuristique seule.")


def _try_load_url_model():
    global _url_model, _url_vocab
    weights_path = os.path.join(URL_MODEL_DIR, "url_lstm.pt")
    vocab_path = os.path.join(URL_MODEL_DIR, "vocab.json")
    if not (os.path.exists(weights_path) and os.path.exists(vocab_path)):
        print(f"[CERBÈRE] Aucun modèle URL trouvé dans {URL_MODEL_DIR} — mode heuristique seul pour la tête URL.")
        return
    try:
        import torch
        from .train_url_lstm import UrlLSTM, VOCAB
        with open(vocab_path, "r", encoding="utf-8") as f:
            _url_vocab = json.load(f)
        model = UrlLSTM(vocab_size=len(VOCAB))
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        _url_model = model
        print(f"[CERBÈRE] Modèle URL chargé depuis {URL_MODEL_DIR}.")
    except ImportError:
        print("[CERBÈRE] PyTorch n'est pas installé — mode heuristique seul pour la tête URL.")
    except Exception as e:
        print(f"[CERBÈRE] Échec du chargement du modèle URL ({e}) — repli sur l'heuristique seule.")


@app.on_event("startup")
def _startup():
    _try_load_text_model()
    _try_load_url_model()


def ml_text_phishing_proba(text: str) -> Optional[float]:
    """Retourne la probabilité de phishing (0-100) selon DistilBERT, ou None si indisponible."""
    if _text_pipeline is None or not text.strip():
        return None
    try:
        out = _text_pipeline(text[:2000])[0]  # liste de {label, score} pour chaque classe
        phishing_entry = next((o for o in out if o["label"] in ("LABEL_1", "1", "phishing")), out[-1])
        return round(phishing_entry["score"] * 100, 1)
    except Exception:
        return None


def ml_url_phishing_proba(url: str) -> Optional[float]:
    if _url_model is None:
        return None
    try:
        import torch
        ids = torch.tensor([encode_url(url)], dtype=torch.long)
        with torch.no_grad():
            logits = _url_model(ids)
            proba = torch.softmax(logits, dim=1)[0, 1].item()
        return round(proba * 100, 1)
    except Exception:
        return None


# ------------------------------------------------------------------ utils
def _indicator_to_dict(ind: H.Indicator, tag: str) -> dict:
    return {"sev": ind.sev, "txt": ind.txt, "tag": tag}


def _verdict(score: float) -> dict:
    v = H.verdict(score)
    return v


# ----------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {
        "status": "ok",
        "text_model_loaded": _text_pipeline is not None,
        "url_model_loaded": _url_model is not None,
    }


@app.post("/analyze/url")
def analyze_url(req: UrlRequest):
    try:
        heur = H.analyze_url_string(req.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible d'analyser cette URL : {e}")
    ml_score = ml_url_phishing_proba(req.url)

    indicators = [_indicator_to_dict(i, "URL") for i in heur.indicators]
    if ml_score is not None:
        sev = "high" if ml_score >= 65 else "med" if ml_score >= 30 else "low"
        indicators.insert(0, {"sev": sev, "txt": f"Modèle LSTM (URL) : probabilité de phishing estimée à <b>{ml_score}%</b>.", "tag": "ML"})
        final_score = 0.5 * heur.score + 0.5 * ml_score
    else:
        final_score = heur.score

    return {
        "mode": "url",
        "final_score": round(final_score, 1),
        "verdict": _verdict(final_score),
        "heads": {
            "nlp": {"score": 0, "indicators": []},
            "url": {"score": round(heur.score, 1), "ml_score": ml_score, "indicators": indicators},
            "headers": {"score": 0, "indicators": []},
        },
    }


@app.post("/analyze/email")
def analyze_email(req: EmailRequest):
    try:
        parsed = P.html_to_text_and_links(req.body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible d'analyser ce message : {e}")
    text_for_nlp = parsed["text"] or req.body

    # Tête NLP (heuristique + modèle si disponible)
    heur_text = H.analyze_text(req.subject, text_for_nlp)
    ml_text_score = ml_text_phishing_proba((req.subject or "") + " " + text_for_nlp)
    nlp_indicators = [_indicator_to_dict(i, "NLP") for i in heur_text.indicators]
    if ml_text_score is not None:
        sev = "high" if ml_text_score >= 65 else "med" if ml_text_score >= 30 else "low"
        nlp_indicators.insert(0, {"sev": sev, "txt": f"Modèle DistilBERT : probabilité de phishing estimée à <b>{ml_text_score}%</b>.", "tag": "ML"})
        nlp_score = 0.5 * heur_text.score + 0.5 * ml_text_score
    else:
        nlp_score = heur_text.score

    # Tête URL : liens HTML (BeautifulSoup) + URLs en texte brut
    candidate_urls = [l["href"] for l in parsed["links"]] + H.extract_urls(req.body)
    candidate_urls = list(dict.fromkeys(candidate_urls))[:5]  # dédoublonnage, limite 5

    url_indicators = []
    url_score = 0.0
    if candidate_urls:
        analyses = [H.analyze_url_string(u) for u in candidate_urls]
        url_score = max(a.score for a in analyses)
        for i, a in enumerate(analyses):
            for ind in a.indicators:
                url_indicators.append({"sev": ind.sev, "txt": f"(lien {i + 1}) {ind.txt}", "tag": "URL"})
        for l in parsed["links"]:
            if l["mismatch"]:
                url_score = min(url_score + 15, 100)
                url_indicators.insert(0, {
                    "sev": "high",
                    "tag": "URL",
                    "txt": f"Lien trompeur : le texte affiché (« {l['text']} ») ne correspond pas à la destination réelle ({l['href']}).",
                })
    else:
        url_indicators.append({"sev": "low", "txt": "Aucun lien détecté dans le corps du message.", "tag": "URL"})

    # Tête en-têtes
    heur_hdr = H.analyze_headers(req.display_name, req.from_addr, req.reply_to)
    hdr_indicators = [_indicator_to_dict(i, "HDR") for i in heur_hdr.indicators]

    final_score = nlp_score * 0.4 + url_score * 0.35 + heur_hdr.score * 0.25

    return {
        "mode": "email",
        "final_score": round(final_score, 1),
        "verdict": _verdict(final_score),
        "heads": {
            "nlp": {"score": round(nlp_score, 1), "ml_score": ml_text_score, "indicators": nlp_indicators},
            "url": {"score": round(url_score, 1), "indicators": url_indicators},
            "headers": {"score": round(heur_hdr.score, 1), "indicators": hdr_indicators},
        },
    }
