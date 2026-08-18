"""
train_url_lstm.py — Entraîne un LSTM caractère-par-caractère pour classer
une URL comme phishing ou légitime, tête "URL" de CERBÈRE.

Contrairement à train_bert.py, ce modèle est entraîné entièrement à partir
de zéro (pas de poids pré-entraînés à télécharger) : c'est l'option la
plus simple à faire tourner hors ligne / dans un environnement contraint,
et elle est bien adaptée aux URLs (séquences de caractères courtes,
pas besoin d'un vocabulaire sémantique).

Prérequis :
  data/url_dataset.csv avec colonnes: url,label (1=phishing, 0=légitime)
  -> Générez-le avec preprocessing.build_url_dataset_csv() à partir d'un
     export PhishTank + d'une liste d'URLs légitimes.

Usage :
  python src/train_url_lstm.py --data data/url_dataset.csv --out models/url_model
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .url_vocab import MAX_LEN, VOCAB, CHAR2IDX, encode_url


class UrlDataset(Dataset):
    def __init__(self, urls, labels):
        self.x = torch.tensor([encode_url(u) for u in urls], dtype=torch.long)
        self.y = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class UrlLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, num_classes)
        )

    def forward(self, x):
        e = self.embed(x)
        out, (h, _) = self.lstm(e)
        h_cat = torch.cat([h[0], h[1]], dim=1)  # dernier état des 2 directions
        return self.fc(h_cat)


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(mode=train)
    total_loss, all_preds, all_labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        all_preds += logits.argmax(dim=1).detach().cpu().tolist()
        all_labels += y.detach().cpu().tolist()
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader.dataset), {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/url_dataset.csv")
    ap.add_argument("--out", default="models/url_model")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(
            f"Introuvable : {args.data}\n"
            "Générez-le via preprocessing.build_url_dataset_csv() à partir "
            "d'un export PhishTank + d'une liste d'URLs légitimes. Voir README.md."
        )

    df = pd.read_csv(args.data).dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    print(f"Jeu de données : {len(df)} URLs "
          f"({(df.label == 1).sum()} phishing / {(df.label == 0).sum()} légitimes)")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = UrlDataset(train_df["url"], train_df["label"])
    val_ds = UrlDataset(val_df["url"], val_df["label"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = UrlLSTM(vocab_size=len(VOCAB)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    os.makedirs(args.out, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_m = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_m = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        print(f"Époque {epoch}/{args.epochs} — train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_f1={val_m['f1']:.4f} val_acc={val_m['accuracy']:.4f}")
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(model.state_dict(), os.path.join(args.out, "url_lstm.pt"))

    with open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"char2idx": CHAR2IDX, "max_len": MAX_LEN}, f, ensure_ascii=False, indent=2)

    print(f"Meilleur F1 (validation) : {best_f1:.4f} — modèle sauvegardé dans {args.out}/url_lstm.pt")


if __name__ == "__main__":
    main()
