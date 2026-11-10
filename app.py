"""
AXON: cómo aprende una red neuronal (DS5343 Visualización de Datos, UTEC 2026-2).

    pip install -r requirements.txt
    python app.py            ->  http://127.0.0.1:5000
"""
import json
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, send_from_directory

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATASETS = ("cifar10", "mnist", "fashion")
FILES = {"nodes.json", "edges.json", "frames.json", "meta.json", "neurons.json", "atlas.png"}

# Autores que aparecen en el pie de la aplicación. Agrega aquí el nombre completo de tu compañera.
AUTHORS = ["Paolo Medrano Terán"
"Gianella Lira"]

app = Flask(__name__)
app.json.ensure_ascii = False


@app.get("/")
def index():
    available = [k for k in DATASETS if (DATA / k / "meta.json").exists()]
    names = {k: json.loads((DATA / k / "meta.json").read_text("utf-8"))["name"] for k in available}
    return render_template("index.html", datasets=available, names=names, authors=AUTHORS)


@app.get("/api/datasets")
def datasets():
    out = []
    for key in DATASETS:
        path = DATA / key / "meta.json"
        if path.exists():
            meta = json.loads(path.read_text("utf-8"))
            out.append({"key": key, "name": meta["name"], "kind": meta["kind"], "partial": meta["partial"]})
    return jsonify(out)


@app.get("/data/<key>/<name>")
def data(key, name):
    if key not in DATASETS or name not in FILES:
        abort(404)
    return send_from_directory(DATA / key, name, max_age=3600)


if __name__ == "__main__":
    app.run(debug=False)
