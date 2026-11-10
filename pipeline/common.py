"""
Funciones comunes del pipeline de AXON.

Todo el cálculo pesado ocurre aquí, en Python. El navegador solo dibuja con D3.
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K = 10                      # vecinos para pureza, conservación y red kNN
PERPLEXITY = 35
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# ---------------------------------------------------------------- vecinos y métricas
def neighbors(values, k=K):
    d = pairwise_distances(values, n_jobs=1)
    np.fill_diagonal(d, np.inf)
    return d.argsort(axis=1, kind="stable")[:, :k]


def describe(embedding, activations, labels):
    """Pureza (vecinos de la misma clase en el espacio original) y conservación en 2D."""
    high = neighbors(activations)
    low = neighbors(embedding)
    per_point = np.array([len(set(a) & set(b)) for a, b in zip(high, low)])
    return {
        "purity": float(np.mean(labels[high] == labels[:, None])),
        "retention": float(per_point.mean() / K),
        "neighbors": high.tolist(),
        "keep": per_point.tolist(),          # 0..10 vecinos conservados por punto
    }


# ---------------------------------------------------------------- t-SNE enlazado
def _tsne(values, init):
    return TSNE(n_components=2, perplexity=PERPLEXITY, max_iter=750, random_state=42,
                init=init, n_jobs=1).fit_transform(values)


def procrustes(embedding, anchor):
    """Rota y refleja `embedding` para alinearlo con `anchor` (mismas observaciones)."""
    e = embedding - embedding.mean(axis=0)
    a = anchor - anchor.mean(axis=0)
    u, _, vt = np.linalg.svd(e.T @ a)
    rotated = e @ (u @ vt)
    scale = np.sqrt((a ** 2).sum() / max((rotated ** 2).sum(), 1e-9))
    return rotated * scale + anchor.mean(axis=0)


def project_anchor(values):
    return _tsne(values, "pca")


def project_linked(values, anchor):
    """t-SNE inicializado con el ancla a escala mínima (no impone grupos falsos) y alineado."""
    a = anchor - anchor.mean(axis=0)
    init = (a / a[:, 0].std() * 1e-4).astype(np.float32)
    return procrustes(_tsne(values, init), anchor)


def normalize_frames(xys):
    """Lleva todas las proyecciones de un dataset a un mismo cuadro [-1, 1] con un único factor."""
    stack = np.concatenate(xys)
    center = (stack.max(axis=0) + stack.min(axis=0)) / 2
    half = np.abs(stack - center).max()
    return [np.round((xy - center) / half, 4) for xy in xys]


# ---------------------------------------------------------------- sonda lineal
def linear_probe(activations, labels):
    """Exactitud de una regresión logística con validación cruzada (separabilidad lineal)."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, C=0.5))
    folds = StratifiedKFold(3, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, activations, labels, cv=folds)
    return float(np.mean(predictions == labels)), predictions.astype(int).tolist()


# ---------------------------------------------------------------- mapa de neuronas
def neuron_map(units, labels, name):
    """units: matriz imágenes × neuronas (activación media de cada neurona en cada imagen)."""
    n = units.shape[1]
    z = (units - units.mean(axis=0)) / (units.std(axis=0) + 1e-6)
    alive = units.std(axis=0) > 1e-6
    profile = z.T                                         # neuronas × imágenes
    if n >= 40:
        xy = TSNE(n_components=2, perplexity=min(15, (n - 1) / 3), random_state=42,
                  init="pca", n_jobs=1).fit_transform(profile)
    else:
        xy = PCA(2, random_state=42).fit_transform(profile)
    xy = (xy - xy.mean(axis=0)) / (np.abs(xy - xy.mean(axis=0)).max() + 1e-9)
    class_means = np.stack([z[labels == c].mean(axis=0) for c in range(10)])   # 10 × n
    preferred = class_means.argmax(axis=0)
    ordered = np.sort(class_means, axis=0)
    selectivity = ordered[-1] - ordered[-2]
    top = np.argsort(-units, axis=0)[:24].T
    return {
        "layer": name,
        "neurons": [
            {"i": int(i), "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4),
             "cls": int(preferred[i]), "sel": round(float(selectivity[i]), 3),
             "alive": bool(alive[i]), "top": top[i].tolist(),
             "profile": [round(float(v), 2) for v in class_means[:, i]]}
            for i in range(n)
        ],
    }


# ---------------------------------------------------------------- red de vecinos (edges)
def knn_network(nbrs, labels, names):
    """Grafo kNN mutuo de la etapa final: aristas, red de clases y fotos puente."""
    nbrs = np.asarray(nbrs)
    g = nx.Graph()
    g.add_nodes_from(range(len(labels)))
    edges = []
    for i, row in enumerate(nbrs):
        for j in row:
            j = int(j)
            a, b = min(i, j), max(i, j)
            if not g.has_edge(a, b):
                g.add_edge(a, b)
                edges.append({"source": a, "target": b,
                              "mutual": bool(i in nbrs[j] and j in nbrs[i]),
                              "inter": bool(labels[a] != labels[b])})
    between = nx.betweenness_centrality(g, k=min(300, len(labels)), seed=42)
    mixed = np.mean(labels[nbrs] != labels[:, None], axis=1)
    score = np.array([between[i] for i in range(len(labels))]) * (0.25 + mixed)
    bridges = [
        {"node": int(i), "betweenness": round(float(between[i]), 5),
         "otherShare": round(float(mixed[i]), 2),
         "towards": int(np.bincount(labels[nbrs[i]][labels[nbrs[i]] != labels[i]],
                                    minlength=10).argmax()) if mixed[i] > 0 else int(labels[i])}
        for i in np.argsort(-score)[:30]
    ]
    pair = np.zeros((10, 10))
    for i, row in enumerate(nbrs):
        for j in row:
            pair[labels[i], labels[j]] += 1
    pair = pair / pair.sum(axis=1, keepdims=True)
    classes = []
    for a in range(10):
        for b in range(a + 1, 10):
            w = (pair[a, b] + pair[b, a]) / 2
            if w > 0.005:
                classes.append({"a": a, "b": b, "w": round(float(w), 4)})
    classes.sort(key=lambda e: -e["w"])
    return {"edges": edges, "bridges": bridges, "classLinks": classes,
            "cohesion": [round(float(pair[c, c]), 4) for c in range(10)], "classNames": names}


# ---------------------------------------------------------------- dificultad por imagen
def difficulty(labels, epoch_predictions, confidence, final_neighbors, epochs):
    preds = np.asarray(epoch_predictions)                 # etapas × imágenes
    flips = (preds[1:] != preds[:-1]).sum(axis=0)
    correct = preds == labels[None, :]
    learned = []
    for i in range(len(labels)):
        col = correct[:, i]
        stable = None
        for s in range(len(col)):
            if col[s:].all():
                stable = epochs[s]
                break
        learned.append(stable)
    other = np.mean(labels[np.asarray(final_neighbors)] != labels[:, None], axis=1)
    wrong = ~correct[-1]
    score = (1 - confidence) + 0.5 * flips / max(flips.max(), 1) + 0.6 * wrong + 0.4 * other
    return [
        {"flips": int(flips[i]), "learned": learned[i], "conf": round(float(confidence[i]), 3),
         "other": round(float(other[i]), 2), "score": round(float(score[i]), 3)}
        for i in range(len(labels))
    ]


# ---------------------------------------------------------------- imágenes
def atlas(images, path, cols=40):
    """Guarda las imágenes en una sola lámina PNG (más liviana que base64 en JSON)."""
    n, h, w = images.shape[:3]
    rows = int(np.ceil(n / cols))
    channels = 3 if images.ndim == 4 else 1
    sheet = np.zeros((rows * h, cols * w, channels), np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = img.reshape(h, w, channels)
    mode = "RGB" if channels == 3 else "L"
    Image.fromarray(sheet.squeeze() if channels == 1 else sheet, mode).save(path, optimize=True)
    return {"cols": cols, "w": w, "h": h}


def save(name, payload):
    folder = DATA / name
    folder.mkdir(parents=True, exist_ok=True)
    for key, value in payload.items():
        (folder / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                                            encoding="utf-8")
    return folder


def pct(v):
    return f"{v * 100:.1f} %".replace(".", ",")


def dec(v, digits=1):
    return f"{v:.{digits}f}".replace(".", ",")
