"""
Descarga los archivos IDX oficiales de MNIST y Fashion-MNIST en data/raw/.

    python -m pipeline.descargar_datos
"""
import urllib.request

from .common import DATA

SOURCES = {
    "mnist": "https://raw.githubusercontent.com/fgnt/mnist/master/{}",
    "fashion": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{}",
}
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

if __name__ == "__main__":
    for key, url in SOURCES.items():
        folder = DATA / "raw" / key
        folder.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            target = folder / name
            if not target.exists():
                print("Descargando", key, name, flush=True)
                urllib.request.urlretrieve(url.format(name), target)
    print("Listo.")
