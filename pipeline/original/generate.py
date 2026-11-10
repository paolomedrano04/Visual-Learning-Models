import argparse
import base64
import hashlib
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from torch import nn
from torch.nn import functional as F


def neighbors(values):
    distances = pairwise_distances(values, n_jobs=1)
    np.fill_diagonal(distances, np.inf)
    return distances.argsort(axis=1, kind="stable")[:, :10]


def describe(embedding, activations, labels):
    high = neighbors(activations)
    low = neighbors(embedding)
    return {
        "purity": float(np.mean(labels[high] == labels[:, None])),
        "retention": float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(high, low)])),
        "neighbors": high.tolist(),
    }


def load_cifar(path):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != "c58f30108f718f92721af3b95e74349a":
            raise ValueError("El archivo no coincide con CIFAR-10 Python oficial.")
    with tarfile.open(path) as archive:
        def batch(name):
            with archive.extractfile(f"cifar-10-batches-py/{name}") as stream:
                data = pickle.load(stream, encoding="bytes")
            return data[b"data"].reshape(-1, 3, 32, 32), np.asarray(data[b"labels"])

        training = [batch(f"data_batch_{i}") for i in range(1, 6)]
        photos, labels = batch("test_batch")
    images, targets = (np.concatenate(values) for values in zip(*training))
    return images, targets, photos, labels


def balanced_sample(labels, count, rng):
    indices = np.concatenate([
        rng.choice(np.flatnonzero(labels == label), count, replace=False) for label in range(10)
    ])
    return rng.permutation(indices)


class CIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(source, target, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(target),
                nn.ReLU(),
                nn.Conv2d(target, target, 3, padding=1, bias=False),
                nn.BatchNorm2d(target),
                nn.ReLU(),
            )
            for source, target in ((3, 32), (32, 64), (64, 96))
        ])
        self.classifier = nn.Sequential(nn.Dropout(.25), nn.Linear(96 * 4, 10))

    def forward(self, values, capture=False):
        representations = []
        for block in self.blocks:
            values = block(values)
            if capture:
                representations.append(F.adaptive_avg_pool2d(values, 2).flatten(1))
        logits = self.classifier(F.adaptive_avg_pool2d(values, 2).flatten(1))
        return (logits, representations) if capture else logits


@torch.inference_mode()
def evaluate(model, images, capture=False):
    model.eval()
    predictions, activations = [], [[], [], []]
    for batch in images.split(256):
        if capture:
            logits, representations = model(batch, capture=True)
            for output, values in zip(activations, representations):
                output.append(values.cpu().numpy())
        else:
            logits = model(batch)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions), [np.concatenate(output) for output in activations] if capture else []


def linked_projections(activations):
    options = dict(n_components=2, perplexity=35, max_iter=750, random_state=42, n_jobs=1)
    anchor = TSNE(init="random", **options).fit_transform(activations[0])
    projections = [anchor]
    for values in activations[1:]:
        embedding = TSNE(init=anchor.copy(), **options).fit_transform(values)
        embedding -= embedding.mean(axis=0)
        left, _, right = np.linalg.svd(embedding.T @ (anchor - anchor.mean(axis=0)))
        projections.append(embedding @ (left @ right) + anchor.mean(axis=0))
    return projections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="Archivo oficial cifar-10-python.tar.gz")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs y threads deben ser positivos")
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images, train_labels, test_images, test_labels = load_cifar(args.archive)
    rng = np.random.default_rng(42)
    development = balanced_sample(train_labels, 2400, rng)
    validation = balanced_sample(train_labels[development], 400, rng)
    train = np.delete(development, validation)
    validation = development[validation]
    visual = validation[balanced_sample(train_labels[validation], 100, rng)]
    pixels = torch.from_numpy(train_images[train]).to(device, dtype=torch.float32) / 255
    mean = pixels.mean(dim=(0, 2, 3), keepdim=True)
    std = pixels.std(dim=(0, 2, 3), keepdim=True)

    def normalize(images):
        values = torch.from_numpy(images).to(device, dtype=torch.float32) / 255
        return ((values - mean) / std).contiguous(memory_format=torch.channels_last)

    x_train = ((pixels - mean) / std).contiguous(memory_format=torch.channels_last)
    y_train = torch.from_numpy(train_labels[train]).to(device)
    x_validation, y_validation = normalize(train_images[validation]), train_labels[validation]
    x_visual, y_visual = normalize(train_images[visual]), train_labels[visual]
    model = CIFARCNN().to(device=device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.0002)
    checkpoints = {1, 2, 4, 8, 12, 16, 24, 32}
    best_score, best_epoch, best_parameters = -1, 0, None

    def snapshot(epoch, score):
        predictions, activations = evaluate(model, x_visual, capture=True)
        return epoch, activations, predictions, score

    initial_predictions, _ = evaluate(model, x_validation)
    snapshots = [snapshot(0, np.mean(initial_predictions == y_validation))]
    print(f"Entrenando CNN en {device}.", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for indices in torch.randperm(len(x_train), device=device).split(128):
            images = x_train[indices]
            flip = torch.rand((len(indices), 1, 1, 1), device=device) < .5
            images = torch.where(flip, images.flip(-1), images)
            offset_x, offset_y = rng.integers(0, 5, size=2)
            images = F.pad(images, (2, 2, 2, 2), mode="reflect")
            images = images[:, :, offset_y:offset_y + 32, offset_x:offset_x + 32]
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.contiguous(memory_format=torch.channels_last))
            F.cross_entropy(logits, y_train[indices]).backward()
            optimizer.step()
        scheduler.step()
        validation_predictions, _ = evaluate(model, x_validation)
        score = np.mean(validation_predictions == y_validation)
        print(f"Época {epoch}: validación {score:.2%}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_parameters = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch in checkpoints:
            snapshots.append(snapshot(epoch, score))
        if epoch - best_epoch >= 10 or epoch == args.epochs:
            if epoch not in checkpoints:
                snapshots.append(snapshot(epoch, score))
            break

    model.load_state_dict(best_parameters)
    selected = snapshot(best_epoch, best_score)
    if all(item[0] != best_epoch for item in snapshots):
        snapshots.append(selected)
        snapshots.sort(key=lambda item: item[0])
    test_predictions, _ = evaluate(model, normalize(test_images))
    test_accuracy = float(np.mean(test_predictions == test_labels))
    print(f"Mejor época: {best_epoch}. Parada: {epoch}. Exactitud de prueba: {test_accuracy:.3%}", flush=True)

    print("Entrenamiento terminado. Calculando proyecciones t-SNE.", flush=True)
    epoch_embeddings = linked_projections([layers[-1] for _, layers, _, _ in snapshots])
    layer_embeddings = linked_projections(selected[1])

    nodes = [
        {"id": int(i), "label": int(train_labels[i]),
         "rgb": base64.b64encode(train_images[i].transpose(1, 2, 0).tobytes()).decode("ascii")}
        for i in visual
    ]
    frames = {"epochs": [], "layers": []}
    edges = []

    def add_frame(mode, title, embedding, activations, predictions, score, epoch):
        frame_id = len(frames[mode])
        frames[mode].append({
            "title": title,
            "xy": embedding.round(5).tolist(),
            "predictions": predictions.tolist(),
            "accuracy": float(np.mean(predictions == y_visual)),
            "validationAccuracy": float(score),
            "epoch": epoch,
            "selected": epoch == best_epoch,
            **describe(embedding, activations, y_visual),
        })
        if frame_id:
            edges.extend(
                {"mode": mode, "id": node["id"], "source": frame_id - 1, "target": frame_id}
                for node in nodes
            )

    for (step, layers, predictions, score), embedding in zip(snapshots, epoch_embeddings):
        add_frame("epochs", f"Época {step}", embedding, layers[-1], predictions, score, step)
    for index, embedding in enumerate(layer_embeddings):
        add_frame("layers", f"Capa oculta {index + 1}", embedding, selected[1][index], selected[2], best_score, best_epoch)

    output = Path(__file__).parent / "static" / "data"
    output.mkdir(parents=True, exist_ok=True)
    meta = {"bestEpoch": best_epoch, "stoppedEpoch": epoch, "testAccuracy": test_accuracy}
    for name, values in [("nodes", nodes), ("frames", frames), ("edges", edges), ("meta", meta)]:
        (output / f"{name}.json").write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        Path(__file__).with_name("model.npz"), mean=mean.cpu().numpy(), std=std.cpu().numpy(),
        epoch=best_epoch, **{key: value.numpy() for key, value in best_parameters.items()},
    )
    for mode, series in frames.items():
        for frame in series:
            print(mode, frame["title"], *(round(frame[key], 3) for key in ("accuracy", "purity", "retention")))


if __name__ == "__main__":
    main()
