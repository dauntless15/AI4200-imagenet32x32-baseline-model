from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from .model import TASKS, ResidualMLP


# First optimization experiment: AdamW with cosine decay; baseline training is still available to the separate comparison tool in analysis.py.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
EPOCHS = {"mnist": 10, "cifar10": 30, "imagenet32": 10}
FINAL_RECIPES = {
    "mnist": {
        "activation": "gelu",
        "normalization": "batchnorm",
        "learning_rate": 1e-3,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": 10,
        "augmentation": False,
    },
    "cifar10": {
        "activation": "gelu",
        "normalization": "batchnorm",
        "learning_rate": 5e-4,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": 15,
        "augmentation": True,
    },
    "imagenet32": {
        "activation": "gelu",
        "normalization": "layernorm",
        "learning_rate": 1e-3,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": 10,
        "augmentation": True,
    },
}


def prepare(X):
    """Convert uint8 images (0-255) to floats in roughly [-1, 1]."""
    return (torch.as_tensor(X).float() / 255.0 - 0.5) / 0.5


def augment_batch(X, task):
    if task == "mnist":
        padding = 2
    elif task in ("cifar10", "imagenet32"):
        padding = 4
    else:
        raise ValueError(f"Unknown task {task!r}; expected one of {tuple(TASKS)}")

    batch_size, _, height, width = X.shape
    padded = F.pad(X, (padding,) * 4, mode="reflect")
    tops = torch.randint(0, 2 * padding + 1, (batch_size,), device=X.device)
    lefts = torch.randint(0, 2 * padding + 1, (batch_size,), device=X.device)
    row_indices = (
        tops[:, None] + torch.arange(height, device=X.device)
    ).view(batch_size, 1, height, 1).expand(
        -1, X.shape[1], -1, padded.shape[-1]
    )
    col_indices = (
        lefts[:, None] + torch.arange(width, device=X.device)
    ).view(batch_size, 1, 1, width).expand(-1, X.shape[1], height, -1)
    cropped = padded.gather(2, row_indices).gather(3, col_indices)

    if task != "mnist":
        flip = torch.rand(batch_size, device=X.device) < 0.5
        cropped = torch.where(flip[:, None, None, None], cropped.flip(-1), cropped)
    return cropped


def _train_model(
    task,
    X_train,
    y_train,
    seed=0,
    device="cuda",
    observer=None,
    train_indices: Optional[Sequence[int]] = None,
    recipe="submission",
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
    epochs=None,
    epoch_callback=None,
    augmentation=False,
    activation="tanh",
    normalization="layernorm",
):
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {tuple(TASKS)}")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of examples")
    if not len(X_train):
        raise ValueError("Training data must contain at least one example")

    torch.manual_seed(seed)
    model = ResidualMLP(
        **TASKS[task],
        activation=activation,
        normalization=normalization,
    ).to(device)

    dataset = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train).long())
    if train_indices is not None:
        dataset = Subset(dataset, train_indices)
    if not len(dataset):
        raise ValueError("The selected training subset must not be empty")

    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=str(device).startswith("cuda"),
        generator=loader_generator,
    )
    loss_fn = nn.CrossEntropyLoss()
    if recipe == "baseline":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        scheduler = None
    elif recipe == "adamw_constant":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = None
    elif recipe == "submission":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS[task]
        )
    else:
        raise ValueError(f"Unknown training recipe {recipe!r}")

    training_epochs = EPOCHS[task] if epochs is None else epochs
    if training_epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if scheduler is not None:
        scheduler.T_max = training_epochs

    history = []
    if observer is not None:
        # Record epoch zero so measurements include the randomly initialized net.
        observer(model, 0, {})

    for epoch in range(training_epochs):
        model.train()
        total_loss = 0.0
        last_update_ratios = {}
        for batch_index, (X_batch, y_batch) in enumerate(loader):
            X_batch = prepare(X_batch.to(device, non_blocking=True))
            if augmentation:
                X_batch = augment_batch(X_batch, task)
            y_batch = y_batch.to(device, non_blocking=True)

            old_weights = {}
            
            capture_update = observer is not None and batch_index == len(loader) - 1
            if capture_update:
                old_weights = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                    if name.endswith(".weight") and parameter.ndim == 2
                }

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if capture_update:
                last_update_ratios = {}
                for name, parameter in model.named_parameters():
                    if name in old_weights:
                        previous = old_weights[name]
                        delta_rms = (parameter.detach() - previous).square().mean().sqrt()
                        weight_rms = previous.square().mean().sqrt()
                        last_update_ratios[name] = (
                            delta_rms / weight_rms.clamp_min(1e-12)
                        ).item()

            total_loss += loss.item() * len(X_batch)

        average_loss = total_loss / len(dataset)
        history.append({"epoch": epoch + 1, "train_loss": average_loss})
        print(f"epoch {epoch + 1}: train loss {average_loss:.4f}")
        if scheduler is not None:
            scheduler.step()
        if observer is not None:
            
            observer(model, epoch + 1, last_update_ratios)
        if epoch_callback is not None:
            epoch_callback(model, epoch + 1, average_loss)

    model.training_history = history
    return model


def train(task, X_train, y_train, seed=0, device="cuda"):
    if task not in FINAL_RECIPES:
        raise ValueError(f"Unknown task {task!r}; expected one of {tuple(FINAL_RECIPES)}")
    return _train_model(
        task,
        X_train,
        y_train,
        seed=seed,
        device=device,
        **FINAL_RECIPES[task],
    )


def predict(model, X):
    device = next(model.parameters()).device
    X = torch.as_tensor(X)
    if len(X) == 0:
        return torch.empty((0, model.head_fc2.out_features))

    # Bound inference memory for the 50,000-image ImageNet32 evaluation set.
    chunks = []
    for start in range(0, len(X), 1024):
        images = prepare(X[start:start + 1024].to(device))
        chunks.append(model.predict(images).cpu())
    return torch.cat(chunks)
