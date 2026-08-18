"""
heuristics.py
Moteur de règles linguistiques et structurelles pour CERBÈRE.
Sert de signal de repli (et d'ensemble) en complément des modèles ML
(train_bert.py pour le texte, train_url_lstm.py pour les URLs).

Ce module ne dépend d'aucun modèle entraîné : il fonctionne immédiatement,
ce qui garantit que l'API reste opérationnelle même sans modèle chargé.
"""
from __future__ import annotations
import re
import unicodedata
from urllib.parse import urlparse
from dataclasses import dataclass, field

URGENCY_WORDS = [
    "urgent", "immédiatement", "immediat", "dans les 24 heures", "24h",
    "derniere chance", "dernier avis", "compte suspendu", "compte bloqué",
    "compte sera fermé", "action requise", "vérifiez maintenant",
    "confirmez immédiatement", "expire aujourd'hui", "expire bientôt",
    "cliquez ici", "cliquez immédiatement", "ne pas ignorer",
    "urgent action required", "verify now", "account suspended",
    "limited time", "act now", "click here immediately",
]
SENSITIVE_WORDS = [
    "mot de passe", "code pin", "numéro de carte", "cvv", "carte bancaire",
    "numéro de sécurité sociale", "identifiants", "code secret",
    "coordonnées bancaires", "rib", "iban", "otp", "code de vérification",
    "password", "credit card", "social security", "ssn",
]
ATTACHMENT_FLAGS = [
    ".exe", ".scr", ".zip", ".js", ".vbs", ".jar",
    "pièce jointe protégée", "facture jointe", "document joint urgent",
]
KNOWN_BRANDS = [
    "paypal", "apple", "microsoft", "google", "amazon", "facebook",
    "instagram", "netflix", "orange", "ooredoo", "tunisie telecom",
    "biat", "attijari", "attijaribank", "stb", "bnp", "la poste",
    "poste tunisienne", "dhl", "fedex", "chronopost", "banque de tunisie",
    "ubci", "zitouna",
]
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work",
    "support", "icu", "monster", "rest",
}
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "ow.ly",
    "cutt.ly", "shorte.st",
}
IP_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
URL_RE = re.compile(r"(https?://[^\s)\]\"']+|www\.[^\s)\]\"']+)", re.IGNORECASE)


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s


def levenshtein(a: str, b: str) -> int:
    a, b = _norm(a), _norm(b)
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[m][n]


def extract_domain(url_or_email: str) -> str:
    s = (url_or_email or "").strip()
    if not s:
        return ""
    if "@" in s and "://" not in s:
        s = s.split("@")[-1]
    if "://" not in s:
        s = "http://" + s
    try:
        return (urlparse(s).hostname or "").lower()
    except Exception:
        return ""


def extract_urls(text: str) -> list[str]:
    found = URL_RE.findall(text or "")
    return [u if u.startswith("http") else "https://" + u for u in found]


def is_ip(host: str) -> bool:
    return bool(IP_RE.match(host or ""))


@dataclass
class Indicator:
    sev: str   # "high" | "med" | "low"
    txt: str
    tag: str = ""


@dataclass
class HeadResult:
    score: float
    indicators: list[Indicator] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def analyze_url_string(raw_url: str) -> HeadResult:
    indicators: list[Indicator] = []
    score = 0.0
    host = ""
    try:
        candidate = raw_url if "://" in raw_url else "https://" + raw_url
        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower()
        full = raw_url

        if is_ip(host):
            score += 22
            indicators.append(Indicator("high", f"L'URL pointe vers une adresse IP brute ({host}) plutôt qu'un nom de domaine."))
        if "@" in full:
            score += 18
            indicators.append(Indicator("high", "La présence du caractère @ peut masquer la vraie destination du lien."))
        if "xn--" in host:
            score += 20
            indicators.append(Indicator("high", "Domaine encodé en punycode — signe possible d'une attaque par caractères homoglyphes."))

        sub = host.split(".") if host else []
        if len(sub) > 3:
            score += 10
            indicators.append(Indicator("med", f"Le domaine comporte {len(sub) - 2} sous-domaines, une technique fréquente pour imiter un site légitime."))

        tld = sub[-1] if sub else ""
        if tld in SUSPICIOUS_TLDS:
            score += 16
            indicators.append(Indicator("high", f"Extension de domaine à risque : .{tld}, très utilisée dans des campagnes de phishing."))

        if not full.startswith("https"):
            score += 8
            indicators.append(Indicator("med", "Le lien n'utilise pas HTTPS, aucune connexion chiffrée n'est garantie."))

        if any(s in host for s in SHORTENERS):
            score += 12
            indicators.append(Indicator("med", "Le lien passe par un raccourcisseur d'URL, ce qui masque la destination réelle."))

        if host.count("-") >= 2:
            score += 8
            indicators.append(Indicator("med", "Le domaine contient plusieurs tirets, souvent utilisés pour imiter une marque."))

        if len(full) > 80:
            score += 6
            indicators.append(Indicator("low", "URL anormalement longue, pouvant dissimuler des paramètres suspects."))

        domain_core = sub[-2] if len(sub) >= 2 else host
        best_brand, best_dist = None, 99
        for b in KNOWN_BRANDS:
            b_core = b.replace(" ", "")
            dist = levenshtein(domain_core, b_core)
            if dist < best_dist:
                best_dist, best_brand = dist, b

        if best_brand and 0 < best_dist <= 2 and _norm(domain_core) != _norm(best_brand).replace(" ", ""):
            score += 26
            indicators.append(Indicator("high", f"Le domaine '{domain_core}' ressemble fortement à la marque '{best_brand}' (distance d'édition : {best_dist}) — typosquatting probable."))
        elif best_brand and best_dist == 0:
            b_core = best_brand.replace(" ", "")
            if not (host.endswith(b_core + ".com") or host.endswith(b_core + ".tn")):
                score += 14
                indicators.append(Indicator("med", f"Le nom '{best_brand}' apparaît dans un domaine qui n'est ni son .com ni son .tn officiel."))
    except Exception:
        score += 5
        indicators.append(Indicator("low", "Format d'URL invalide ou non standard."))

    if not indicators:
        indicators.append(Indicator("low", "Aucun indicateur structurel suspect détecté sur cette URL."))

    return HeadResult(score=min(score, 100), indicators=indicators, meta={"host": host})


def analyze_text(subject: str, body: str) -> HeadResult:
    indicators: list[Indicator] = []
    score = 0.0
    full = _norm((subject or "") + " " + (body or ""))

    urgency_hits = [w for w in URGENCY_WORDS if _norm(w) in full]
    if urgency_hits:
        score += min(10 * len(urgency_hits), 30)
        indicators.append(Indicator(
            "high" if len(urgency_hits) >= 3 else "med",
            f"Formulations de pression/urgence détectées ({len(urgency_hits)}) : « {', '.join(urgency_hits[:3])} »",
        ))

    sensitive_hits = [w for w in SENSITIVE_WORDS if _norm(w) in full]
    if sensitive_hits:
        score += min(12 * len(sensitive_hits), 30)
        indicators.append(Indicator("high", f"Demande d'informations sensibles repérée : « {', '.join(sensitive_hits[:3])} »"))

    attach_hits = [w for w in ATTACHMENT_FLAGS if _norm(w) in full]
    if attach_hits:
        score += 12
        indicators.append(Indicator("med", "Mention de pièce jointe à risque ou inhabituelle dans le message."))

    exclam = (body or "").count("!")
    if exclam >= 3:
        score += 6
        indicators.append(Indicator("low", f"Ponctuation excessive ({exclam} points d'exclamation), signe fréquent de manipulation émotionnelle."))

    caps_words = [w for w in (body or "").split() if len(w) > 3 and w.isupper()]
    if len(caps_words) >= 3:
        score += 6
        indicators.append(Indicator("low", "Usage répété de MOTS EN MAJUSCULES pour attirer l'attention."))

    if not indicators:
        indicators.append(Indicator("low", "Aucun schéma linguistique typique de l'ingénierie sociale détecté dans le texte."))

    return HeadResult(score=min(score, 100), indicators=indicators)


def analyze_headers(display_name: str, from_addr: str, reply_to: str = "") -> HeadResult:
    indicators: list[Indicator] = []
    score = 0.0
    from_domain = extract_domain(from_addr or "")
    reply_domain = extract_domain(reply_to) if reply_to else ""
    name_norm = _norm(display_name or "")

    brand_in_name = next((b for b in KNOWN_BRANDS if _norm(b) in name_norm), None)
    if brand_in_name and from_domain:
        brand_core = brand_in_name.replace(" ", "")
        domain_labels = from_domain.split(".")
        domain_core = domain_labels[-2] if len(domain_labels) >= 2 else from_domain
        # comparaison stricte du libellé de second niveau (pas une simple
        # sous-chaîne) : "paypal" est présent dans "paypal-alerte.tk" mais
        # ce n'est PAS le domaine officiel de la marque -> doit être détecté.
        if _norm(domain_core) != _norm(brand_core):
            score += 28
            indicators.append(Indicator("high", f"Le nom affiché cite '{brand_in_name}' mais l'adresse réelle ({from_domain}) n'appartient pas à ce domaine."))

    if reply_to and reply_domain and from_domain and reply_domain != from_domain:
        score += 20
        indicators.append(Indicator("high", f"L'adresse de réponse ({reply_domain}) diffère du domaine d'expédition ({from_domain}) — technique classique de détournement de réponse."))

    if from_domain and is_ip(from_domain):
        score += 15
        indicators.append(Indicator("high", "L'expéditeur utilise une adresse IP comme nom de domaine."))

    if from_domain:
        tld = from_domain.split(".")[-1]
        if tld in SUSPICIOUS_TLDS:
            score += 14
            indicators.append(Indicator("med", f"Domaine d'expédition avec extension à risque : .{tld}."))

    if not from_domain:
        indicators.append(Indicator("low", "Adresse d'expéditeur non renseignée ou invalide — analyse limitée."))
    elif not indicators:
        indicators.append(Indicator("low", "Aucune incohérence détectée entre le nom affiché, l'expéditeur et l'adresse de réponse."))

    return HeadResult(score=min(score, 100), indicators=indicators, meta={"from_domain": from_domain})


def verdict(score: float) -> dict:
    if score < 30:
        return {"label": "E-MAIL SÛR", "level": "safe"}
    if score < 65:
        return {"label": "SUSPECT", "level": "suspect"}
    return {"label": "PHISHING PROBABLE", "level": "phishing"}
