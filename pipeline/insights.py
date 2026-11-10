"""
Hallazgos automáticos de las cuatro preguntas, calculados desde los JSON ya generados.
Así los números del texto siempre coinciden con los de la visualización.

    python -m pipeline.insights
"""
import json

import numpy as np

from .common import DATA, dec, pct

FEMININE = {"Camiseta", "Sandalia", "Camisa", "Zapatilla"}


def article(key, name):
    """Cómo nombrar una imagen de esa clase dentro de una oración."""
    if key == "cifar10":
        return f"una foto de {name.lower()}"
    if key == "mnist":
        return f"un {name}"
    return f"{'una' if name in FEMININE else 'un'} {name.lower()}"


def load(key):
    folder = DATA / key
    return {name: json.loads((folder / f"{name}.json").read_text("utf-8"))
            for name in ("meta", "frames", "nodes", "edges")}


def build(key):
    d = load(key)
    meta, frames, nodes, edges = d["meta"], d["frames"], d["nodes"], d["edges"]
    names = meta["classes"]
    row, col = meta["rowLayer"], meta["colEpoch"]
    stages = meta["epochs"]
    layers = meta["layers"]
    cell = lambda s, l: frames.get(f"{s}:{l}")
    labels = np.array([n["label"] for n in nodes])

    # P1 épocas
    pur_e = [cell(s, row)["purity"] for s in range(len(stages))]
    val_e = [s["val"] for s in stages]
    jumps = np.diff(pur_e)
    j = int(np.argmax(jumps))
    best = max(val_e)
    plateau = next(s for s in range(len(stages)) if best - val_e[s] <= 0.015)
    epochs = [
        f"Sin entrenar (época 0), el {pct(pur_e[0])} de los vecinos ya comparte clase, frente al 10 % "
        f"que daría el azar.",
        f"El mayor salto ocurre entre la {stages[j]['label'].lower()} y la {stages[j + 1]['label'].lower()}: "
        f"la pureza sube {dec(jumps[j] * 100)} puntos (de {pct(pur_e[j])} a {pct(pur_e[j + 1])}).",
        f"Desde la {stages[plateau]['label'].lower()} la validación queda a menos de 1,5 puntos de su mejor valor "
        f"({pct(best)}): lo que sigue es ajuste fino.",
    ]

    # P2 capas
    pur_l = [cell(col, l)["purity"] for l in range(len(layers))]
    prb_l = [cell(col, l)["probeAcc"] for l in range(len(layers))]
    lj = np.diff(pur_l)
    k = int(np.argmax(lj))
    layer_text = [
        f"En píxeles, el {pct(pur_l[0])} de los vecinos comparte clase.",
        f"El mayor salto de pureza ocurre al pasar a {layers[k + 1]} (+{dec(lj[k] * 100)} puntos).",
        f"Una regresión lineal sobre cada capa acierta {pct(prb_l[0])} con píxeles y {pct(prb_l[-2])} "
        f"con {layers[-2]}.",
    ]
    if cell(0, 0) and cell(0, row):
        p0 = [cell(0, l)["purity"] for l in range(len(layers))]
        if p0[-1] < p0[0]:
            layer_text.append(
                f"Sin entrenar, cada capa pierde información: la pureza cae de {pct(p0[0])} en píxeles a "
                f"{pct(p0[-1])} en la salida. Entrenar invierte esa tendencia.")

    # P3 confusiones y puentes
    final = cell(col, row)
    pred = np.array(final["pred"])
    conf = np.zeros((10, 10), int)
    for t, p in zip(labels, pred):
        conf[t, p] += 1
    off = conf + conf.T
    np.fill_diagonal(off, 0)
    a, b = np.unravel_index(off.argmax(), off.shape)
    link = edges["classLinks"][0]
    coh = np.array(edges["cohesion"])
    bridge = edges["bridges"][0]
    bn = nodes[bridge["node"]]
    confusion = [
        f"La pareja más mezclada es {names[link['a']]} y {names[link['b']]}: comparten el "
        f"{pct(link['w'])} de sus vecinos.",
        f"El modelo confunde {names[a]} con {names[b]} en {off[a, b]} de estas 1.000 imágenes.",
        f"La clase más compacta es {names[int(coh.argmax())]} ({pct(coh.max())} de vecinos propios) y la más "
        f"dispersa, {names[int(coh.argmin())]} ({pct(coh.min())}).",
        f"La foto puente principal es {article(key, names[bn['label']])} con el {pct(bridge['otherShare'])} de "
        f"vecinos de otra clase, sobre todo {names[bridge['towards']]}.",
    ]

    # P4 imágenes difíciles
    wrong = int(np.sum(pred != labels))
    never = sum(1 for n in nodes if n["learned"] is None)
    flips = np.array([n["flips"] for n in nodes])
    learned = np.array([n["learned"] if n["learned"] is not None else stages[-1]["epoch"] + 1 for n in nodes])
    hard = int(np.argmax([learned[labels == c].mean() for c in range(10)]))
    hard_text = [
        f"{wrong} de 1.000 imágenes terminan mal clasificadas y {never} nunca se estabilizan en la respuesta correcta.",
        f"{int(np.sum(flips >= 3))} imágenes cambian de predicción 3 veces o más; la más indecisa cambia "
        f"{int(flips.max())} veces.",
        f"{names[hard]} es la clase que más tarda en aprenderse.",
    ]
    meta["insights"] = {"epochs": epochs, "layers": layer_text, "confusion": confusion, "hard": hard_text}
    meta["topPair"] = [int(link["a"]), int(link["b"])]
    meta["articles"] = [article(key, nm) for nm in names]
    (DATA / key / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), "utf-8")
    return meta["insights"]


if __name__ == "__main__":
    for key in ("cifar10", "mnist", "fashion"):
        if (DATA / key / "meta.json").exists():
            for group, lines in build(key).items():
                for line in lines:
                    print(f"[{key}:{group}] {line}")
