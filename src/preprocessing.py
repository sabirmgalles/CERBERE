"""
preprocessing.py
- Parsing HTML des e-mails avec BeautifulSoup (extraction de texte, liens,
  détection de liens "masqués" où le texte affiché ≠ la vraie destination).
- Fonctions de construction des jeux de données pour l'entraînement
  (train_bert.py / train_url_lstm.py).

Les datasets externes (PhishTank, Enron Spam, corpus Nazario) ne sont pas
inclus dans ce dépôt : ce sandbox n'a pas accès à ces domaines. Placez les
fichiers téléchargés dans data/ (voir README.md) puis lancez ces scripts
sur votre machine.
"""
from __future__ import annotations
import re
import csv
import glob
import os
from bs4 import BeautifulSoup


def html_to_text_and_links(html_or_text: str) -> dict:
    """
    Retourne {'text': str, 'links': [{'text': str, 'href': str, 'mismatch': bool}]}
    Fonctionne aussi bien sur du HTML que sur du texte brut (BeautifulSoup
    ne trouvera simplement aucun tag dans ce dernier cas).
    """
    soup = BeautifulSoup(html_or_text or "", "html.parser")
    links = []
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        visible = a.get_text(strip=True)
        if not href:
            continue
        mismatch = bool(visible) and visible.startswith("http") and visible not in href and href not in visible
        links.append({"text": visible, "href": href, "mismatch": mismatch})

    text = soup.get_text(separator=" ", strip=True)
    if not text:
        # pas de balises -> texte brut tel quel
        text = re.sub(r"\s+", " ", html_or_text or "").strip()

    # liens en texte brut (non-HTML) capturés séparément par heuristics.extract_urls
    return {"text": text, "links": links}


def load_enron_ham(enron_dir: str) -> list[str]:
    """
    Attend l'arborescence classique du corpus Enron-Spam
    (dossiers .../ham/*.txt). Retourne la liste des corps de message.
    Téléchargement : https://www2.aueb.gr/users/ion/data/enron-spam/
    """
    texts = []
    for path in glob.glob(os.path.join(enron_dir, "**", "ham", "*.txt"), recursive=True):
        try:
            with open(path, "r", encoding="latin-1") as f:
                texts.append(f.read())
        except OSError:
            continue
    return texts


def load_phishing_corpus(phishing_dir: str) -> list[str]:
    """
    Attend un dossier de fichiers texte d'e-mails de phishing, par ex. le
    corpus Nazario (couramment utilisé en complément d'Enron pour la
    classification de texte). Un fichier par e-mail.
    """
    texts = []
    for path in glob.glob(os.path.join(phishing_dir, "*")):
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="latin-1") as f:
                    texts.append(f.read())
            except OSError:
                continue
    return texts


def build_text_dataset_csv(enron_dir: str, phishing_dir: str, out_csv: str) -> int:
    """Construit data/text_dataset.csv avec colonnes text,label (1=phishing, 0=légitime)."""
    ham = load_enron_ham(enron_dir)
    phish = load_phishing_corpus(phishing_dir)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for t in ham:
            w.writerow([html_to_text_and_links(t)["text"][:4000], 0])
        for t in phish:
            w.writerow([html_to_text_and_links(t)["text"][:4000], 1])
    return len(ham) + len(phish)


def load_phishtank_urls(phishtank_csv: str) -> list[str]:
    """
    Charge les URLs depuis un export PhishTank (colonne 'url').
    Téléchargement : https://phishtank.org/developer_info.php
    """
    urls = []
    with open(phishtank_csv, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row.get("url") or row.get("URL")
            if u:
                urls.append(u.strip())
    return urls


def build_url_dataset_csv(phishtank_csv: str, benign_urls_csv: str, out_csv: str) -> int:
    """
    Construit data/url_dataset.csv avec colonnes url,label (1=phishing, 0=légitime).
    benign_urls_csv : liste d'URLs légitimes, une par ligne (par ex. un
    échantillon du top Tranco/Majestic, ou vos propres logs).
    """
    phishing = load_phishtank_urls(phishtank_csv)
    benign = []
    with open(benign_urls_csv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                benign.append(line)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "label"])
        for u in phishing:
            w.writerow([u, 1])
        for u in benign:
            w.writerow([u, 0])
    return len(phishing) + len(benign)
