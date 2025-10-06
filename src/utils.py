"""
Funciones auxiliares de visualización y métricas.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_history(hist):
    if hist is None: return
    h = hist.history
    plt.figure()
    plt.plot(h.get("loss",[]), label="loss")
    if "val_loss" in h: plt.plot(h["val_loss"], label="val_loss")
    if "accuracy" in h: plt.plot(h["accuracy"], label="acc")
    if "val_accuracy" in h: plt.plot(h["val_accuracy"], label="val_acc")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("metric"); plt.title("Training curves")
    plt.show()

def make_windows(series, window=32):
    X, y = [], []
    for i in range(len(series)-window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    X = np.array(X)[...,None]
    y = np.array(y)[...,None]
    return X, y


# Resultados en CSV
import csv, os, time, json

def append_summary_row(csv_path, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp","module","model","dataset","accuracy","f1","loss","epochs","params","train_time_s","notes"])
        if write_header: w.writeheader()
        row_out = {k: row.get(k, "") for k in ["timestamp","module","model","dataset","accuracy","f1","loss","epochs","params","train_time_s","notes"]}
        w.writerow(row_out)

def now_ts():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- RAG logging ---
def save_rag_example(path, query, retrieved, response, correct=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    recs = []
    for p, score, snippet in retrieved:
        recs.append({"path": str(p), "score": float(score), "snippet": snippet})
    payload = {
        "timestamp": now_ts(),
        "query": query,
        "retrieved": recs,
        "response": response,
        "correct": bool(correct) if correct is not None else None
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
