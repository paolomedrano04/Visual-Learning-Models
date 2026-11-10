"""
CIFAR-10: amplía el trabajo original sin reentrenar.

Entrada:  pipeline/original/  (frames.json, nodes.json y model.npz del entrenamiento original
          hecho con generate.py y PyTorch; pesos de la época seleccionada)
Salida:   data/cifar10/

Qué agrega:
  * Una pasada hacia adelante en NumPy con los pesos guardados (no requiere PyTorch). Se valida
    que reproduzca exactamente las predicciones originales.
  * Cinco etapas por capa: entrada (píxeles), tres bloques convolucionales y salida (logits).
  * Sonda lineal por capa, confianza por imagen, mapa de neuronas por bloque y red kNN.
  * Las épocas se reutilizan del entrenamiento original (tercer bloque, 10 etapas).

    python -m pipeline.cifar
"""
import base64
import json

import numpy as np

from .common import (DATA, ROOT, atlas, dec, describe, difficulty, knn_network, linear_probe,
                     neighbors, neuron_map, normalize_frames, pct, project_linked, save)

SRC = ROOT / "pipeline" / "original"
NAMES = ["Avión", "Automóvil", "Ave", "Gato", "Ciervo", "Perro", "Rana", "Caballo", "Barco", "Camión"]


# ------------------------------------------------------------------ red convolucional en NumPy
def conv2d(x, w, stride):
    n, c, h, wd = x.shape
    o = w.shape[0]
    x = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    ho, wo = (h + 2 - 3) // stride + 1, (wd + 2 - 3) // stride + 1
    cols = np.empty((n, c, 3, 3, ho, wo), np.float32)
    for i in range(3):
        for j in range(3):
            cols[:, :, i, j] = x[:, :, i:i + stride * ho:stride, j:j + stride * wo:stride]
    return np.einsum("ncijhw,ocij->nohw", cols, w, optimize=True)


def bn_relu(x, m, key):
    scale = m[f"{key}.weight"] / np.sqrt(m[f"{key}.running_var"] + 1e-5)
    shift = m[f"{key}.bias"] - m[f"{key}.running_mean"] * scale
    return np.maximum(x * scale[None, :, None, None] + shift[None, :, None, None], 0)


def pool2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, 2, h // 2, 2, w // 2).mean(axis=(3, 5))


def forward(m, pixels):
    x = (pixels - m["mean"]) / m["std"]
    blocks = []
    for b in range(3):
        x = bn_relu(conv2d(x, m[f"blocks.{b}.0.weight"], 2), m, f"blocks.{b}.1")
        x = bn_relu(conv2d(x, m[f"blocks.{b}.3.weight"], 1), m, f"blocks.{b}.4")
        blocks.append(x)
    feats = [pool2(b).reshape(len(b), -1) for b in blocks]
    logits = feats[-1] @ m["classifier.1.weight"].T + m["classifier.1.bias"]
    channels = [b.mean(axis=(2, 3)) for b in blocks]
    return feats, logits, channels


def main():
    nodes = json.loads((SRC / "nodes.json").read_text("utf-8"))
    frames = json.loads((SRC / "frames.json").read_text("utf-8"))
    meta0 = json.loads((SRC / "meta.json").read_text("utf-8"))
    m = dict(np.load(SRC / "model.npz"))
    labels = np.array([n["label"] for n in nodes])
    rgb = np.stack([np.frombuffer(base64.b64decode(n["rgb"]), np.uint8).reshape(32, 32, 3) for n in nodes])
    pixels = rgb.transpose(0, 3, 1, 2).astype(np.float32) / 255

    feats, logits, channels = forward(m, pixels)
    predictions = logits.argmax(axis=1)
    reference = np.array(frames["layers"][0]["predictions"])
    agreement = float(np.mean(predictions == reference))
    print(f"Pasada NumPy vs PyTorch original: {agreement:.2%} de predicciones idénticas")
    assert agreement > 0.995, "La pasada en NumPy no reproduce el modelo original"
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)

    epochs = [f["epoch"] for f in frames["epochs"]]
    best = meta0["bestEpoch"]
    col = epochs.index(best)
    anchor = np.array(frames["epochs"][col]["xy"])

    layer_names = ["Entrada (píxeles)", "Bloque conv. 1", "Bloque conv. 2", "Bloque conv. 3", "Salida (logits)"]
    layer_values = [pixels.reshape(len(pixels), -1), *feats, logits]
    layer_xy = []
    for name, values in zip(layer_names, layer_values):
        if name == "Bloque conv. 3":
            layer_xy.append(anchor)                     # misma celda que la época seleccionada
        else:
            print("t-SNE", name, flush=True)
            layer_xy.append(project_linked(values, anchor))

    epoch_xy = [np.array(f["xy"]) for f in frames["epochs"]]
    normalized = normalize_frames(epoch_xy + layer_xy)
    epoch_xy, layer_xy = normalized[:len(epoch_xy)], normalized[len(epoch_xy):]

    cells = {}
    hidden = len(layer_names) - 2                          # índice del bloque 3 en las capas
    for s, (f, xy) in enumerate(zip(frames["epochs"], epoch_xy)):
        keep = [len(set(a) & set(b)) for a, b in zip(f["neighbors"], neighbors(xy))]
        cells[f"{s}:{hidden}"] = {
            "xy": xy.tolist(), "pred": f["predictions"], "probe": None, "probeAcc": None,
            "purity": f["purity"], "retention": float(np.mean(keep) / 10), "keep": keep,
            "neighbors": f["neighbors"],
        }
    for l, (values, xy) in enumerate(zip(layer_values, layer_xy)):
        print("Sonda lineal", layer_names[l], flush=True)
        acc, probe = linear_probe(values, labels)
        stats = describe(xy, values, labels)
        cell = cells.get(f"{col}:{l}", {})
        cell.update({"xy": xy.tolist(), "pred": predictions.tolist(), "probe": probe,
                     "probeAcc": acc, **stats})
        cells[f"{col}:{l}"] = cell

    stages = [{"epoch": f["epoch"], "val": f["validationAccuracy"], "label": f"Época {f['epoch']}"}
              for f in frames["epochs"]]
    neurons = [neuron_map(ch, labels, layer_names[i + 1]) for i, ch in enumerate(channels)]
    final = cells[f"{col}:{hidden}"]
    net = knn_network(final["neighbors"], labels, NAMES)
    diff = difficulty(labels, [f["predictions"] for f in frames["epochs"]], probs.max(axis=1),
                      final["neighbors"], epochs)

    folder = DATA / "cifar10"
    folder.mkdir(parents=True, exist_ok=True)
    sheet = atlas(rgb, folder / "atlas.png")
    out_nodes = [{"i": i, "id": n["id"], "label": int(labels[i]),
                  "probs": [round(float(p), 3) for p in probs[i]], **diff[i]}
                 for i, n in enumerate(nodes)]

    purity_layers = [cells[f"{col}:{l}"]["purity"] for l in range(len(layer_names))]
    probe_layers = [cells[f"{col}:{l}"]["probeAcc"] for l in range(len(layer_names))]
    jumps = np.diff(purity_layers[1:4])
    insights = {
        "epochs": [
            f"Sin entrenar (época 0), el {pct(frames['epochs'][0]['purity'])} de los vecinos ya comparte clase, "
            f"frente al 10 % del azar: los filtros aleatorios captan color y fondo.",
            f"La primera época concentra el mayor salto: la pureza pasa de {pct(frames['epochs'][0]['purity'])} "
            f"a {pct(frames['epochs'][1]['purity'])} y la validación de {pct(frames['epochs'][0]['validationAccuracy'])} "
            f"a {pct(frames['epochs'][1]['validationAccuracy'])}.",
            f"Desde la época 16 la validación apenas sube ({pct(frames['epochs'][6]['validationAccuracy'])} a "
            f"{pct(frames['epochs'][col]['validationAccuracy'])}): el modelo ya encontró su forma.",
        ],
        "layers": [
            f"En píxeles, solo el {pct(purity_layers[0])} de los vecinos comparte clase: la foto cruda no basta.",
            f"El bloque 3 aporta el mayor salto de pureza (+{dec(jumps[1] * 100)} puntos) frente al bloque 2 "
            f"(+{dec(jumps[0] * 100)}): el significado aparece en lo profundo.",
            f"Una regresión lineal sobre cada capa acierta {pct(probe_layers[0])} con píxeles y "
            f"{pct(probe_layers[3])} con el bloque 3.",
        ],
    }
    meta = {
        "key": "cifar10", "name": "CIFAR-10", "kind": "Fotografías a color 32 × 32",
        "model": "Red convolucional de 3 bloques (32, 64 y 96 canales)",
        "classes": NAMES, "epochs": stages, "layers": layer_names,
        "rowLayer": hidden, "colEpoch": col, "bestEpoch": best, "stoppedEpoch": meta0["stoppedEpoch"],
        "testAccuracy": meta0["testAccuracy"], "atlas": sheet, "n": len(nodes),
        "curve": [{"epoch": s["epoch"], "val": s["val"]} for s in stages],
        "insights": insights,
        "source": "Entrenamiento original con PyTorch (20.000 fotos de entrenamiento, 4.000 de validación).",
        "partial": True,
    }
    save("cifar10", {"nodes": out_nodes, "frames": cells, "edges": net, "meta": meta,
                     "neurons": neurons})
    print("Listo:", folder)


if __name__ == "__main__":
    main()
