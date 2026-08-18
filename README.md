# CERBÈRE — Backend (API + entraînement)

Ce dossier contient la partie "vrai ML" de CERBÈRE, en complément du
frontend `cerbere-detection-phishing.html` déjà livré :

```
cerbere-backend/
├── requirements.txt
├── data/                    ← placez vos datasets ici (non fournis)
├── models/                  ← modèles entraînés (générés par vous)
└── src/
    ├── heuristics.py        # règles (port Python du moteur JS du frontend)
    ├── preprocessing.py     # parsing HTML/BeautifulSoup + construction des CSV
    ├── train_bert.py        # fine-tuning DistilBERT — tête "NLP" (texte)
    ├── train_url_lstm.py    # LSTM caractère — tête "URL"
    └── api.py               # backend FastAPI qui sert le tout
```

## 0. Statut vérifié dans ce sandbox (transparence totale)

Le réseau de cet environnement de développement est actuellement bloqué
même vers pypi.org (`403 host_not_allowed`) — impossible d'y installer
fastapi/torch/transformers ni de télécharger un dataset. Voici précisément
ce qui a été **réellement exécuté et validé ici**, et ce qui ne l'a pas été :

| Composant | Statut |
|---|---|
| `heuristics.py` (règles) | ✅ 36 tests unitaires exécutés, tous passants (`tests/`) |
| `preprocessing.py` (BeautifulSoup) | ✅ testé, y compris détection de liens trompeurs |
| `url_vocab.py` (encodage URL) | ✅ testé, indépendant de torch |
| Syntaxe de tous les fichiers `.py` | ✅ compilée sans erreur |
| `api.py` (FastAPI, endpoints) | ⚠️ relu et durci (validation, erreurs, dégradation sans torch), **jamais exécuté** — fastapi non installable ici |
| `train_bert.py` / `train_url_lstm.py` | ⚠️ syntaxiquement corrects, **jamais entraînés** — torch non installable ici |
| Bug réel trouvé et corrigé grâce aux tests | ✅ `analyze_headers` ratait les domaines type `paypal-alerte.tk` (la marque en sous-chaîne du domaine passait inaperçue) — corrigé, couvert par un test de non-régression |

**À faire sur votre machine** (accès réseau complet) : `pip install -r
requirements.txt`, lancer `python -m unittest discover -s tests` pour
confirmer que tout passe aussi chez vous, puis suivre les étapes 2 à 4
ci-dessous avec de vraies données.


## Pourquoi les datasets ne sont pas inclus

Ce sandbox de développement n'a accès qu'à un nombre limité de domaines
(pypi.org, github.com, npmjs.com...) — **pas** à phishtank.org, huggingface.co
ni aux archives d'Enron/Kaggle. Les scripts sont donc écrits et testés pour
la syntaxe et la logique, mais l'entraînement réel doit se faire sur votre
machine, avec accès internet complet.

## 1. Installation

```bash
python -m venv venv && source venv/bin/activate   # ou l'équivalent Windows
pip install -r requirements.txt
python -m unittest discover -s tests -v   # doit afficher 36/36 OK
```

## 2. Récupérer les données

| Dataset | Usage | Source |
|---|---|---|
| Enron-Spam (dossiers `ham/`) | e-mails légitimes | https://www2.aueb.gr/users/ion/data/enron-spam/ |
| Corpus de phishing (ex. Nazario) | e-mails de phishing | recherchez "Nazario phishing corpus" |
| PhishTank (export CSV) | URLs de phishing | https://phishtank.org/developer_info.php |
| Liste d'URLs légitimes | ex. échantillon du top Tranco | https://tranco-list.eu/ |

Puis construisez les CSV d'entraînement :

```python
from src.preprocessing import build_text_dataset_csv, build_url_dataset_csv

build_text_dataset_csv("data/enron", "data/phishing_corpus", "data/text_dataset.csv")
build_url_dataset_csv("data/phishtank.csv", "data/benign_urls.csv", "data/url_dataset.csv")
```

## 3. Entraîner les modèles

```bash
# Tête NLP (DistilBERT multilingue, fine-tuné sur les corps d'e-mail)
python src/train_bert.py --data data/text_dataset.csv --out models/text_model --epochs 3

# Tête URL (LSTM caractère, entraîné from scratch — pas de téléchargement de poids)
python src/train_url_lstm.py --data data/url_dataset.csv --out models/url_model --epochs 8
```

Sans GPU, `train_bert.py` reste utilisable mais plus lent ; `train_url_lstm.py`
est volontairement léger et tourne bien sur CPU.

## 4. Lancer l'API

```bash
uvicorn src.api:app --reload --port 8000
```

- `GET /health` → indique si les modèles ML sont chargés
- `POST /analyze/email` → `{display_name, from_addr, reply_to, subject, body}`
- `POST /analyze/url` → `{url}`

**L'API fonctionne même sans modèle entraîné** : les trois têtes heuristiques
(`heuristics.py`) tournent toujours et servent de repli. Dès qu'un modèle est
présent dans `models/text_model/` ou `models/url_model/`, sa prédiction est
automatiquement chargée au démarrage et fusionnée au score (50/50) — aucune
modification de code n'est nécessaire.

## 5. Connecter le frontend

Dans `cerbere-detection-phishing.html`, un sélecteur **Local / API** a été
ajouté en haut du module d'analyse. En mode API, le frontend appelle
`http://localhost:8000/analyze/email` et `/analyze/url` en `fetch()` ; si
l'API n'est pas joignable, un message l'indique et le mode local reste
disponible en repli.

## Sécurité — avant toute mise en production

- `allow_origins=["*"]` dans `api.py` est réglé pour la démo ; restreignez-le
  à votre propre domaine.
- Aucune authentification n'est en place sur les routes — à ajouter avant
  toute exposition publique.
- Le contenu des e-mails analysés n'est pas persisté par l'API telle quelle,
  mais vérifiez vos obligations RGPD si vous journalisez les requêtes.
