"""
MNIST y Fashion-MNIST: perceptrón multicapa entrenado en NumPy (sin PyTorch), como en el paper.

Guarda las activaciones de TODAS las capas en cada época de control, así que estos datasets
tienen la matriz completa época × capa.

    python -m pipeline.mlp mnist
    python -m pipeline.mlp fashion

Datos: archivos IDX oficiales en data/raw/<dataset>/ (train-images-idx3-ubyte.gz, etc.).
"""
import gzip
import sys

import numpy as np

from .common import (DATA, atlas, dec, describe, difficulty, knn_network, linear_probe,
                     neuron_map, normalize_frames, pct, project_anchor, project_linked, save)

SETS = {
    "mnist": {"name": "MNIST", "kind": "Dígitos manuscritos 28 × 28",
              "classes": [str(i) for i in range(10)]},
    "fashion": {"name": "Fashion-MNIST", "kind": "Prendas de vestir 28 × 28",
                "classes": ["Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
                            "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]},
}
HIDDEN = (256, 128, 64, 32)
EPOCHS = 20
CHECKPOINTS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 20]


def read_idx(path, offset):
    with gzip.open(path) as f:
        return np.frombuffer(f.read(), np.uint8, offset=offset)


def load(key):
    raw = DATA / "raw" / key
    x = read_idx(raw / "train-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28)
    y = read_idx(raw / "train-labels-idx1-ubyte.gz", 8).astype(int)
    xt = read_idx(raw / "t10k-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28)
    yt = read_idx(raw / "t10k-labels-idx1-ubyte.gz", 8).astype(int)
    return x, y, xt, yt


def balanced(labels, per_class, rng, exclude=None):
    pool = np.ones(len(labels), bool)
    if exclude is not None:
        pool[exclude] = False
    return rng.permutation(np.concatenate([
        rng.choice(np.flatnonzero((labels == c) & pool), per_class, replace=False) for c in range(10)]))


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, x):
        acts = [x]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ w + b
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return acts                                         # entrada, ocultas..., logits

    def step(self, x, y, lr=1e-3, l2=1e-4):
        acts = self.forward(x)
        logits = acts[-1]
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        loss = -np.log(p[np.arange(len(y)), y] + 1e-9).mean()
        grad = p
        grad[np.arange(len(y)), y] -= 1
        grad /= len(y)
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.insert(0, acts[i].T @ grad + l2 * self.W[i])
            gb.insert(0, grad.sum(axis=0))
            if i:
                grad = (grad @ self.W[i].T) * (acts[i] > 0)
        self.t += 1
        for j, (param, g) in enumerate(zip(self.W + self.b, gW + gb)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * g
            self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
            mh = self.m[j] / (1 - 0.9 ** self.t)
            vh = self.v[j] / (1 - 0.999 ** self.t)
            param -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss

    def predict(self, x):
        return np.concatenate([self.forward(chunk)[-1].argmax(axis=1) for chunk in np.array_split(x, 20)])

    def snapshot(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def load(self, snap):
        self.W, self.b = [w.copy() for w in snap[0]], [b.copy() for b in snap[1]]


def main(key):
    spec = SETS[key]
    rng = np.random.default_rng(42)
    x, y, xt, yt = load(key)
    train = balanced(y, 2000, rng)
    val = balanced(y, 500, rng, exclude=train)
    visual = balanced(yt, 100, rng)
    flat = lambda a: a.reshape(len(a), -1).astype(np.float32) / 255
    xtr, ytr, xva, yva = flat(x[train]), y[train], flat(x[val]), y[val]
    xvis, yvis = flat(xt[visual]), yt[visual]

    net = MLP([784, *HIDDEN, 10], rng)
    snaps, curve = {0: net.snapshot()}, [{"epoch": 0, "val": float(np.mean(net.predict(xva) == yva)), "loss": None}]
    best, best_val = 0, curve[0]["val"]
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(xtr))
        losses = [net.step(xtr[idx], ytr[idx]) for idx in np.array_split(order, len(order) // 128)]
        val_acc = float(np.mean(net.predict(xva) == yva))
        curve.append({"epoch": epoch, "val": val_acc, "loss": float(np.mean(losses))})
        print(f"{spec['name']} época {epoch}: pérdida {np.mean(losses):.3f}, validación {val_acc:.2%}", flush=True)
        if epoch in CHECKPOINTS:
            snaps[epoch] = net.snapshot()
        if val_acc > best_val:
            best, best_val = epoch, val_acc
            snaps[epoch] = net.snapshot()
    keep = sorted(set(CHECKPOINTS) | {best})
    net.load(snaps[best])
    test_acc = float(np.mean(net.predict(flat(xt)) == yt))
    print(f"Mejor época {best} ({best_val:.2%}); prueba {test_acc:.2%}", flush=True)

    layer_names = ["Entrada (píxeles)", *[f"Oculta {i + 1} ({h})" for i, h in enumerate(HIDDEN)], "Salida (logits)"]
    acts, preds, probs = {}, {}, {}
    for e in keep:
        net.load(snaps[e])
        a = net.forward(xvis)
        acts[e] = a
        preds[e] = a[-1].argmax(axis=1)
        z = np.exp(a[-1] - a[-1].max(axis=1, keepdims=True))
        probs[e] = z / z.sum(axis=1, keepdims=True)

    row_layer = len(HIDDEN)                                  # última capa oculta
    col = keep.index(best)
    anchor = project_anchor(acts[best][row_layer])
    xys, order = [], []
    for s, e in enumerate(keep):
        for l in range(len(layer_names)):
            print(f"t-SNE época {e}, {layer_names[l]}", flush=True)
            xys.append(anchor if (e == best and l == row_layer) else project_linked(acts[e][l], anchor))
            order.append((s, l))
    xys = normalize_frames(xys)

    cells = {}
    for (s, l), xy in zip(order, xys):
        e = keep[s]
        acc, probe = linear_probe(acts[e][l], yvis)
        cells[f"{s}:{l}"] = {"xy": xy.tolist(), "pred": preds[e].tolist(), "probe": probe,
                             "probeAcc": acc, **describe(xy, acts[e][l], yvis)}
        print(f"celda {s}:{l} pureza {cells[f'{s}:{l}']['purity']:.3f} sonda {acc:.3f}", flush=True)

    net.load(snaps[best])
    neurons = [neuron_map(acts[best][l], yvis, layer_names[l]) for l in range(1, len(layer_names) - 1)]
    final = cells[f"{col}:{row_layer}"]
    net_info = knn_network(final["neighbors"], yvis, spec["classes"])
    diff = difficulty(yvis, [preds[e] for e in keep], probs[best].max(axis=1), final["neighbors"], keep)

    folder = DATA / key
    folder.mkdir(parents=True, exist_ok=True)
    sheet = atlas(xt[visual], folder / "atlas.png")
    nodes = [{"i": i, "id": int(visual[i]), "label": int(yvis[i]),
              "probs": [round(float(p), 3) for p in probs[best][i]], **diff[i]} for i in range(len(visual))]

    pur_e = [cells[f"{s}:{row_layer}"]["purity"] for s in range(len(keep))]
    pur_l = [cells[f"{col}:{l}"]["purity"] for l in range(len(layer_names))]
    prb_l = [cells[f"{col}:{l}"]["probeAcc"] for l in range(len(layer_names))]
    jumps = np.diff(pur_l)
    big = int(np.argmax(jumps[:-1]))
    insights = {
        "epochs": [
            f"Sin entrenar, la última capa oculta ya agrupa el {pct(pur_e[0])} de los vecinos por clase.",
            f"Tras una sola época la pureza llega a {pct(pur_e[1])} y la validación a {pct(curve[1]['val'])}.",
            f"La mejor validación ({pct(best_val)}) llega en la época {best}; la prueba oficial da {pct(test_acc)}.",
        ],
        "layers": [
            f"En píxeles, el {pct(pur_l[0])} de los vecinos ya comparte clase: este dataset es más fácil que CIFAR-10.",
            f"El mayor salto de pureza ocurre al pasar a {layer_names[big + 1]} (+{dec(jumps[big] * 100)} puntos).",
            f"Una regresión lineal acierta {pct(prb_l[0])} con píxeles y {pct(prb_l[row_layer])} con la última capa oculta.",
        ],
    }
    meta = {
        "key": key, "name": spec["name"], "kind": spec["kind"],
        "model": "Perceptrón multicapa 784, 256, 128, 64, 32, 10 (ReLU, Adam)",
        "classes": spec["classes"], "layers": layer_names,
        "epochs": [{"epoch": e, "val": curve[e]["val"], "label": f"Época {e}"} for e in keep],
        "rowLayer": row_layer, "colEpoch": col, "bestEpoch": best, "stoppedEpoch": EPOCHS,
        "testAccuracy": test_acc, "atlas": sheet, "n": len(visual), "curve": curve,
        "insights": insights, "partial": False,
        "source": "20.000 imágenes de entrenamiento y 5.000 de validación (partición oficial de entrenamiento); "
                  "las 1.000 visualizadas vienen de la partición de prueba.",
    }
    save(key, {"nodes": nodes, "frames": cells, "edges": net_info, "meta": meta, "neurons": neurons})
    print("Listo:", folder)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mnist")
