"""Optional ImageNet32 training-dynamics and residual-branch ablation tools."""

import argparse
import csv
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from .train import _train_model, prepare


CSV_FIELDS = (
    "epoch",
    "kind",
    "name",
    "activation_variance",
    "branch_to_stream_ratio",
    "gradient_rms",
    "update_to_weight_ratio",
    "tanh_saturation_fraction",
)


def _rms(values):
    return values.detach().float().square().mean().sqrt().item()


def measure_dynamics(model, X_fixed, y_fixed, epoch, update_ratios):
    """Measure block activations and per-linear-layer gradients on one batch."""
    device = next(model.parameters()).device
    was_training = model.training
    images = prepare(torch.as_tensor(X_fixed).to(device))
    labels = torch.as_tensor(y_fixed).long().to(device)
    block_values = {}
    saturation = {}
    handles = []

    for index, block in enumerate(model.blocks):
        def capture_block(_module, inputs, output, block_index=index):
            _ = _module
            stream = inputs[0].detach().float()
            # Subtract the skip input to recover the residual branch contribution.
            branch = output.detach().float() - stream
            block_values[block_index] = {
                "activation_variance": output.detach().float().var(
                    dim=0, unbiased=False
                ).mean().item(),
                "branch_to_stream_ratio": _rms(branch) / max(_rms(stream), 1e-12),
            }

        def capture_pre_activation(_module, _inputs, output, block_index=index):
            _ = _module, _inputs
            saturation[block_index] = (
                output.detach().abs() > 2
            ).float().mean().item()

        handles.append(block.register_forward_hook(capture_block))
        if model.activation_name == "tanh":
            handles.append(
                block.fc1.register_forward_hook(capture_pre_activation)
            )

    try:
        # Eval mode fixes normalization behavior; gradients are still enabled
        # explicitly so layer-gradient measurements remain available.
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(images)
            nn.functional.cross_entropy(logits, labels).backward()

        rows = []
        for index in range(len(model.blocks)):
            rows.append({
                "epoch": epoch,
                "kind": "block",
                "name": f"blocks.{index}",
                **block_values[index],
                "tanh_saturation_fraction": saturation.get(index),
            })

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                gradient_rms = (
                    _rms(module.weight.grad)
                    if module.weight.grad is not None
                    else float("nan")
                )
                rows.append({
                    "epoch": epoch,
                    "kind": "linear",
                    "name": name,
                    "gradient_rms": gradient_rms,
                    "update_to_weight_ratio": update_ratios.get(name + ".weight"),
                })
        return rows
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def _validation_accuracy(model, dataset, indices, device, batch_size=1024):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=str(device).startswith("cuda"),
    )
    correct = 0
    count = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = prepare(images.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            count += len(labels)
    return correct / count


def measure_branch_ablations(model, X, y, validation_indices, device):
    """Evaluate each branch-disabled model on the same held-out validation set."""
    dataset = TensorDataset(torch.as_tensor(X), torch.as_tensor(y).long())
    full_accuracy = _validation_accuracy(
        model, dataset, validation_indices, device
    )
    results = []
    for index, block in enumerate(model.blocks):
        # Replace only this block's output with its input, retaining the skip.
        def skip_branch(_module, inputs, _output):
            _ = _module, _output
            return inputs[0]

        handle = block.register_forward_hook(skip_branch)
        try:
            accuracy = _validation_accuracy(
                model, dataset, validation_indices, device
            )
        finally:
            handle.remove()
        results.append({
            "block": index,
            "accuracy": accuracy,
            "accuracy_drop": full_accuracy - accuracy,
        })
    return full_accuracy, results


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Install the optional analysis dependencies with "
            "`pip install -e '.[analysis]'` to create plots."
        ) from error
    return plt


def plot_dynamics(rows, output_path):
    plt = _pyplot()
    available_epochs = sorted({int(row["epoch"]) for row in rows})
    epochs = sorted({
        available_epochs[0],
        available_epochs[min(1, len(available_epochs) - 1)],
        available_epochs[len(available_epochs) // 2],
        available_epochs[-1],
    })
    block_rows = [row for row in rows if row["kind"] == "block"]
    linear_rows = [row for row in rows if row["kind"] == "linear"]
    metrics = (
        ("activation_variance", "Activation variance", block_rows, "blocks"),
        ("branch_to_stream_ratio", "Branch / stream RMS", block_rows, "blocks"),
        (
            "tanh_saturation_fraction",
            "Tanh saturation fraction (only for tanh)",
            block_rows,
            "blocks",
        ),
        ("gradient_rms", "Weight gradient RMS", linear_rows, "layers"),
        ("update_to_weight_ratio", "Update / weight RMS", linear_rows, "layers"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(13, 13))
    block_count = max(
        (int(row["name"].split(".")[1]) for row in block_rows),
        default=-1,
    ) + 1

    def layer_depth(name):
        if name == "input_layer":
            return -1
        if name.startswith("blocks."):
            _, block_index, layer_name = name.split(".")
            return int(block_index) + (0.5 if layer_name == "fc2" else 0)
        if name == "head_fc1":
            return block_count
        if name == "head_fc2":
            return block_count + 0.5
        return block_count + 1

    for axis, (key, title, selected, _) in zip(axes.flat, metrics):
        for epoch in epochs:
            epoch_rows = [
                row for row in selected if int(row["epoch"]) == epoch
            ]
            values = []
            for row in epoch_rows:
                raw_value = row.get(key)
                if raw_value is None or raw_value == "":
                    continue
                x = (
                    int(row["name"].split(".")[1])
                    if row["kind"] == "block"
                    else layer_depth(row["name"])
                )
                values.append((x, float(raw_value)))
            if values:
                axis.plot(
                    [item[0] for item in values],
                    [item[1] for item in values],
                    marker=".",
                    label=f"epoch {epoch}",
                )
        if not axis.lines and key == "tanh_saturation_fraction":
            axis.text(
                0.5,
                0.5,
                "Not applicable for GELU",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_title(title)
        axis.set_xlabel("Block/layer depth")
        axis.set_ylabel(title)
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend()
    axes.flat[0].set_yscale("log")
    axes.flat[-1].axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_ablations(results, output_path):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        [row["block"] for row in results],
        [row["accuracy_drop"] for row in results],
        marker="o",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Residual block depth")
    axis.set_ylabel("Validation accuracy drop")
    axis.set_title("Effect of disabling one residual branch")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_training_curves(histories, output_path):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, history in histories.items():
        axis.plot(
            [row["epoch"] for row in history],
            [row["train_loss"] for row in history],
            label=label,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training cross-entropy loss")
    axis.set_title("Training loss by recipe")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_optimizer_comparison(
    task,
    X_train,
    y_train,
    output_dir="analysis",
    device="cpu",
    seed=0,
    validation_fraction=0.1,
):
    """Compare the starter SGD recipe with the submission recipe on train-only validation."""
    if task not in ("mnist", "cifar10", "imagenet32"):
        raise ValueError(f"Unknown task {task!r}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(X_train) != len(y_train) or len(y_train) < 2:
        raise ValueError("Training arrays must have matching examples")

    import numpy as np

    # Keep the held-out validation indices out of both training runs.
    permutation = np.random.default_rng(seed).permutation(len(y_train))
    validation_count = max(1, int(len(y_train) * validation_fraction))
    validation_indices = permutation[:validation_count].tolist()
    training_indices = permutation[validation_count:].tolist()
    dataset = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train).long())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    results = []
    for recipe, label in (("baseline", "SGD baseline"), ("submission", "AdamW + cosine")):
        start = time.perf_counter()
        model = _train_model(
            task,
            X_train,
            y_train,
            seed=seed,
            device=device,
            train_indices=training_indices,
            recipe=recipe,
        )
        elapsed = time.perf_counter() - start
        histories[label] = model.training_history
        results.append({
            "recipe": label,
            "validation_accuracy": _validation_accuracy(
                model, dataset, validation_indices, device
            ),
            "training_seconds": elapsed,
        })

    _write_csv(
        output_dir / f"{task}_optimizer_comparison.csv",
        [
            {"recipe": recipe, **row}
            for recipe, history in histories.items()
            for row in history
        ],
        ("recipe", "epoch", "train_loss"),
    )
    _write_csv(
        output_dir / f"{task}_validation_results.csv",
        results,
        ("recipe", "validation_accuracy", "training_seconds"),
    )
    plot_training_curves(
        histories, output_dir / f"{task}_training_curves.png"
    )
    return results


def run_schedule_comparison(
    task,
    X_train,
    y_train,
    output_dir="analysis",
    device="cpu",
    seed=0,
    validation_fraction=0.1,
):
    """Compare constant-rate AdamW with AdamW plus cosine decay."""
    if task not in ("mnist", "cifar10", "imagenet32"):
        raise ValueError(f"Unknown task {task!r}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(X_train) != len(y_train) or len(y_train) < 2:
        raise ValueError("Training arrays must have matching examples")

    import numpy as np

    # Both runs get exactly the same training and validation examples.
    permutation = np.random.default_rng(seed).permutation(len(y_train))
    validation_count = max(1, int(len(y_train) * validation_fraction))
    validation_indices = permutation[:validation_count].tolist()
    training_indices = permutation[validation_count:].tolist()
    dataset = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train).long())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    results = []
    recipes = (
        ("adamw_constant", "AdamW constant lr"),
        ("submission", "AdamW cosine"),
    )
    for recipe, label in recipes:
        start = time.perf_counter()
        model = _train_model(
            task,
            X_train,
            y_train,
            seed=seed,
            device=device,
            train_indices=training_indices,
            recipe=recipe,
        )
        elapsed = time.perf_counter() - start
        histories[label] = model.training_history
        results.append({
            "recipe": label,
            "validation_accuracy": _validation_accuracy(
                model, dataset, validation_indices, device
            ),
            "training_seconds": elapsed,
        })

    _write_csv(
        output_dir / f"{task}_schedule_comparison.csv",
        [
            {"recipe": recipe, **row}
            for recipe, history in histories.items()
            for row in history
        ],
        ("recipe", "epoch", "train_loss"),
    )
    _write_csv(
        output_dir / f"{task}_schedule_validation_results.csv",
        results,
        ("recipe", "validation_accuracy", "training_seconds"),
    )
    plot_training_curves(
        histories, output_dir / f"{task}_schedule_training_curves.png"
    )
    return results


def run_learning_rate_comparison(
    task,
    X_train,
    y_train,
    output_dir="analysis",
    device="cpu",
    seed=0,
    validation_fraction=0.1,
    learning_rates=(5e-4, 1e-3, 2e-3),
):
    """Compare AdamW cosine runs at multiple learning rates on one fixed split."""
    if task not in ("mnist", "cifar10", "imagenet32"):
        raise ValueError(f"Unknown task {task!r}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(X_train) != len(y_train) or len(y_train) < 2:
        raise ValueError("Training arrays must have matching examples")
    if not learning_rates or any(rate <= 0 for rate in learning_rates):
        raise ValueError("learning_rates must contain positive values")

    import numpy as np

    permutation = np.random.default_rng(seed).permutation(len(y_train))
    validation_count = max(1, int(len(y_train) * validation_fraction))
    validation_indices = permutation[:validation_count].tolist()
    training_indices = permutation[validation_count:].tolist()
    dataset = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train).long())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    results = []
    for learning_rate in learning_rates:
        label = f"AdamW cosine lr={learning_rate:g}"
        start = time.perf_counter()
        model = _train_model(
            task,
            X_train,
            y_train,
            seed=seed,
            device=device,
            train_indices=training_indices,
            recipe="submission",
            learning_rate=learning_rate,
        )
        elapsed = time.perf_counter() - start
        histories[label] = model.training_history
        results.append({
            "learning_rate": learning_rate,
            "validation_accuracy": _validation_accuracy(
                model, dataset, validation_indices, device
            ),
            "final_train_loss": model.training_history[-1]["train_loss"],
            "training_seconds": elapsed,
        })

    _write_csv(
        output_dir / f"{task}_learning_rate_comparison.csv",
        [
            {"recipe": recipe, **row}
            for recipe, history in histories.items()
            for row in history
        ],
        ("recipe", "epoch", "train_loss"),
    )
    _write_csv(
        output_dir / f"{task}_learning_rate_validation_results.csv",
        results,
        ("learning_rate", "validation_accuracy", "final_train_loss", "training_seconds"),
    )
    plot_training_curves(
        histories, output_dir / f"{task}_learning_rate_training_curves.png"
    )
    return results


def run_weight_decay_comparison(
    task,
    X_train,
    y_train,
    output_dir="analysis",
    device="cpu",
    seed=0,
    validation_fraction=0.1,
    weight_decays=(0.0, 1e-5, 1e-4, 1e-3),
    learning_rate=None,
):
    """Compare AdamW weight decay values with the other training settings fixed."""
    if task not in ("mnist", "cifar10", "imagenet32"):
        raise ValueError(f"Unknown task {task!r}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(X_train) != len(y_train) or len(y_train) < 2:
        raise ValueError("Training arrays must have matching examples")
    if not weight_decays or any(value < 0 for value in weight_decays):
        raise ValueError("weight_decays must contain non-negative values")

    import numpy as np
    from .train import LEARNING_RATE

    if learning_rate is None:
        learning_rate = {"mnist": 1e-3, "cifar10": 5e-4, "imagenet32": LEARNING_RATE}[task]

    permutation = np.random.default_rng(seed).permutation(len(y_train))
    validation_count = max(1, int(len(y_train) * validation_fraction))
    validation_indices = permutation[:validation_count].tolist()
    training_indices = permutation[validation_count:].tolist()
    dataset = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train).long())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    results = []
    for weight_decay in weight_decays:
        label = f"AdamW cosine wd={weight_decay:g}"
        start = time.perf_counter()
        model = _train_model(
            task,
            X_train,
            y_train,
            seed=seed,
            device=device,
            train_indices=training_indices,
            recipe="submission",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        elapsed = time.perf_counter() - start
        histories[label] = model.training_history
        results.append({
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "validation_accuracy": _validation_accuracy(
                model, dataset, validation_indices, device
            ),
            "final_train_loss": model.training_history[-1]["train_loss"],
            "training_seconds": elapsed,
        })

    _write_csv(
        output_dir / f"{task}_weight_decay_comparison.csv",
        [
            {"recipe": recipe, **row}
            for recipe, history in histories.items()
            for row in history
        ],
        ("recipe", "epoch", "train_loss"),
    )
    _write_csv(
        output_dir / f"{task}_weight_decay_validation_results.csv",
        results,
        (
            "weight_decay",
            "learning_rate",
            "validation_accuracy",
            "final_train_loss",
            "training_seconds",
        ),
    )
    plot_training_curves(
        histories, output_dir / f"{task}_weight_decay_training_curves.png"
    )
    return results


def run_batch_size_comparison(
    task,
    X_train,
    y_train,
    output_dir="analysis",
    device="cpu",
    seed=0,
    validation_fraction=0.1,
    batch_sizes=(128, 256, 512),
    learning_rate=None,
):
    """Compare mini-batch sizes with AdamW, cosine decay, and LR held fixed."""
    if task not in ("mnist", "cifar10", "imagenet32"):
        raise ValueError(f"Unknown task {task!r}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(X_train) != len(y_train) or len(y_train) < 2:
        raise ValueError("Training arrays must have matching examples")
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive values")

    import numpy as np
    from .train import LEARNING_RATE

    if learning_rate is None:
        learning_rate = {"mnist": 1e-3, "cifar10": 5e-4, "imagenet32": LEARNING_RATE}[task]

    permutation = np.random.default_rng(seed).permutation(len(y_train))
    validation_count = max(1, int(len(y_train) * validation_fraction))
    validation_indices = permutation[:validation_count].tolist()
    training_indices = permutation[validation_count:].tolist()
    dataset = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train).long())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    results = []
    for batch_size in batch_sizes:
        label = f"AdamW cosine batch={batch_size}"
        start = time.perf_counter()
        model = _train_model(
            task,
            X_train,
            y_train,
            seed=seed,
            device=device,
            train_indices=training_indices,
            recipe="submission",
            learning_rate=learning_rate,
            weight_decay=1e-4,
            batch_size=batch_size,
        )
        elapsed = time.perf_counter() - start
        histories[label] = model.training_history
        results.append({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_accuracy": _validation_accuracy(
                model, dataset, validation_indices, device
            ),
            "final_train_loss": model.training_history[-1]["train_loss"],
            "training_seconds": elapsed,
        })

    _write_csv(
        output_dir / f"{task}_batch_size_comparison.csv",
        [
            {"recipe": recipe, **row}
            for recipe, history in histories.items()
            for row in history
        ],
        ("recipe", "epoch", "train_loss"),
    )
    _write_csv(
        output_dir / f"{task}_batch_size_validation_results.csv",
        results,
        (
            "batch_size",
            "learning_rate",
            "validation_accuracy",
            "final_train_loss",
            "training_seconds",
        ),
    )
    plot_training_curves(
        histories, output_dir / f"{task}_batch_size_training_curves.png"
    )
    return results


def run_training_duration_comparison(
    task,
    X_train,
    y_train,
    output_dir="analysis",
    device="cpu",
    seed=0,
    validation_fraction=0.1,
    epoch_budgets=None,
):
    """Compare training duration using validation accuracy at each epoch."""
    if task not in ("mnist", "cifar10", "imagenet32"):
        raise ValueError(f"Unknown task {task!r}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(X_train) != len(y_train) or len(y_train) < 2:
        raise ValueError("Training arrays must have matching examples")

    import numpy as np
    from .train import EPOCHS

    if epoch_budgets is None:
        epoch_budgets = {
            "mnist": (5, 10, 20),
            "cifar10": (15, 30, 45),
            "imagenet32": (5, 10, 20),
        }[task]
    if not epoch_budgets or any(epochs <= 0 for epochs in epoch_budgets):
        raise ValueError("epoch_budgets must contain positive epoch counts")
    if len(set(epoch_budgets)) != len(epoch_budgets):
        raise ValueError("epoch_budgets must not contain duplicates")

    permutation = np.random.default_rng(seed).permutation(len(y_train))
    validation_count = max(1, int(len(y_train) * validation_fraction))
    validation_indices = permutation[:validation_count].tolist()
    training_indices = permutation[validation_count:].tolist()
    dataset = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train).long())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_rows = []
    summary_rows = []
    learning_rate = {"mnist": 1e-3, "cifar10": 5e-4, "imagenet32": 1e-3}[task]
    max_epochs = max(epoch_budgets)

    for budget in sorted(epoch_budgets):
        def record_epoch(model, epoch, train_loss):
            validation_accuracy = _validation_accuracy(
                model, dataset, validation_indices, device
            )
            epoch_rows.append({
                "run_epochs": budget,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_accuracy": validation_accuracy,
            })

        start = time.perf_counter()
        model = _train_model(
            task,
            X_train,
            y_train,
            seed=seed,
            device=device,
            train_indices=training_indices,
            recipe="submission",
            learning_rate=learning_rate,
            weight_decay=1e-4,
            batch_size=256,
            epochs=budget,
            epoch_callback=record_epoch,
        )
        elapsed = time.perf_counter() - start
        summary_rows.append({
            "epochs": budget,
            "learning_rate": learning_rate,
            "validation_accuracy": epoch_rows[-1]["validation_accuracy"],
            "final_train_loss": model.training_history[-1]["train_loss"],
            "training_seconds": elapsed,
        })

    _write_csv(
        output_dir / f"{task}_training_duration_by_epoch.csv",
        epoch_rows,
        ("run_epochs", "epoch", "train_loss", "validation_accuracy"),
    )
    _write_csv(
        output_dir / f"{task}_training_duration_results.csv",
        summary_rows,
        ("epochs", "learning_rate", "validation_accuracy", "final_train_loss", "training_seconds"),
    )
    _plot_training_duration(epoch_rows, output_dir / f"{task}_training_duration.png")
    return summary_rows


def _plot_training_duration(rows, output_path):
    plt = _pyplot()
    budgets = sorted({row["run_epochs"] for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for budget in budgets:
        selected = [row for row in rows if row["run_epochs"] == budget]
        axes[0].plot(
            [row["epoch"] for row in selected],
            [row["train_loss"] for row in selected],
            label=f"{budget} epochs",
        )
        axes[1].plot(
            [row["epoch"] for row in selected],
            [row["validation_accuracy"] for row in selected],
            label=f"{budget} epochs",
        )
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_augmentation_comparison(
    task,
    X_train,
    y_train,
    output_dir="analysis",
    device="cpu",
    seed=0,
    validation_fraction=0.1,
):
    """Compare baseline training against task-specific spatial augmentation."""
    if task not in ("mnist", "cifar10", "imagenet32"):
        raise ValueError(f"Unknown task {task!r}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(X_train) != len(y_train) or len(y_train) < 2:
        raise ValueError("Training arrays must have matching examples")

    import numpy as np
    from .train import EPOCHS

    permutation = np.random.default_rng(seed).permutation(len(y_train))
    validation_count = max(1, int(len(y_train) * validation_fraction))
    validation_indices = permutation[:validation_count].tolist()
    training_indices = permutation[validation_count:].tolist()
    dataset = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train).long())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    learning_rate = {"mnist": 1e-3, "cifar10": 5e-4, "imagenet32": 1e-3}[task]
    histories = {}
    results = []

    for use_augmentation, label in (
        (False, "No augmentation"),
        (True, "Spatial augmentation"),
    ):
        start = time.perf_counter()
        model = _train_model(
            task,
            X_train,
            y_train,
            seed=seed,
            device=device,
            train_indices=training_indices,
            recipe="submission",
            learning_rate=learning_rate,
            weight_decay=1e-4,
            batch_size=256,
            epochs=EPOCHS[task],
            augmentation=use_augmentation,
        )
        elapsed = time.perf_counter() - start
        histories[label] = model.training_history
        results.append({
            "augmentation": use_augmentation,
            "learning_rate": learning_rate,
            "epochs": EPOCHS[task],
            "validation_accuracy": _validation_accuracy(
                model, dataset, validation_indices, device
            ),
            "final_train_loss": model.training_history[-1]["train_loss"],
            "training_seconds": elapsed,
        })

    _write_csv(
        output_dir / f"{task}_augmentation_comparison.csv",
        [
            {"recipe": recipe, **row}
            for recipe, history in histories.items()
            for row in history
        ],
        ("recipe", "epoch", "train_loss"),
    )
    _write_csv(
        output_dir / f"{task}_augmentation_validation_results.csv",
        results,
        (
            "augmentation",
            "learning_rate",
            "epochs",
            "validation_accuracy",
            "final_train_loss",
            "training_seconds",
        ),
    )
    plot_training_curves(
        histories, output_dir / f"{task}_augmentation_training_curves.png"
    )
    return results


def run_activation_comparison(
    task,
    X_train,
    y_train,
    output_dir="analysis",
    device="cpu",
    seed=0,
    validation_fraction=0.1,
    activations=("tanh", "relu", "gelu"),
):
    """Compare activation functions with architecture and training settings fixed."""
    if task not in ("mnist", "cifar10", "imagenet32"):
        raise ValueError(f"Unknown task {task!r}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(X_train) != len(y_train) or len(y_train) < 2:
        raise ValueError("Training arrays must have matching examples")
    if not activations or any(name not in ("tanh", "relu", "gelu") for name in activations):
        raise ValueError("activations must be selected from tanh, relu, and gelu")

    import numpy as np
    from .train import EPOCHS

    permutation = np.random.default_rng(seed).permutation(len(y_train))
    validation_count = max(1, int(len(y_train) * validation_fraction))
    validation_indices = permutation[:validation_count].tolist()
    training_indices = permutation[validation_count:].tolist()
    dataset = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train).long())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    learning_rate = {"mnist": 1e-3, "cifar10": 5e-4, "imagenet32": 1e-3}[task]
    histories = {}
    results = []

    for activation in activations:
        label = activation
        start = time.perf_counter()
        model = _train_model(
            task,
            X_train,
            y_train,
            seed=seed,
            device=device,
            train_indices=training_indices,
            recipe="submission",
            learning_rate=learning_rate,
            weight_decay=1e-4,
            batch_size=256,
            epochs=EPOCHS[task],
            augmentation=True,
            activation=activation,
        )
        elapsed = time.perf_counter() - start
        histories[label] = model.training_history
        results.append({
            "activation": activation,
            "learning_rate": learning_rate,
            "epochs": EPOCHS[task],
            "augmentation": True,
            "validation_accuracy": _validation_accuracy(
                model, dataset, validation_indices, device
            ),
            "final_train_loss": model.training_history[-1]["train_loss"],
            "training_seconds": elapsed,
        })

    _write_csv(
        output_dir / f"{task}_activation_comparison.csv",
        [
            {"activation": activation, **row}
            for activation, history in histories.items()
            for row in history
        ],
        ("activation", "epoch", "train_loss"),
    )
    _write_csv(
        output_dir / f"{task}_activation_validation_results.csv",
        results,
        (
            "activation",
            "learning_rate",
            "epochs",
            "augmentation",
            "validation_accuracy",
            "final_train_loss",
            "training_seconds",
        ),
    )
    plot_training_curves(
        histories, output_dir / f"{task}_activation_training_curves.png"
    )
    return results


def run_normalization_comparison(
    task,
    X_train,
    y_train,
    output_dir="analysis",
    device="cpu",
    seed=0,
    validation_fraction=0.1,
    normalizations=("layernorm", "batchnorm"),
):
    """Compare LayerNorm and BatchNorm with the rest of the recipe fixed."""
    if task not in ("mnist", "cifar10", "imagenet32"):
        raise ValueError(f"Unknown task {task!r}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(X_train) != len(y_train) or len(y_train) < 2:
        raise ValueError("Training arrays must have matching examples")
    if not normalizations or any(
        name not in ("layernorm", "batchnorm") for name in normalizations
    ):
        raise ValueError("normalizations must be layernorm and/or batchnorm")

    import numpy as np
    from .train import EPOCHS

    permutation = np.random.default_rng(seed).permutation(len(y_train))
    validation_count = max(1, int(len(y_train) * validation_fraction))
    validation_indices = permutation[:validation_count].tolist()
    training_indices = permutation[validation_count:].tolist()
    dataset = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train).long())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    learning_rate = {"mnist": 1e-3, "cifar10": 5e-4, "imagenet32": 1e-3}[task]
    histories = {}
    results = []

    for normalization in normalizations:
        start = time.perf_counter()
        model = _train_model(
            task,
            X_train,
            y_train,
            seed=seed,
            device=device,
            train_indices=training_indices,
            recipe="submission",
            learning_rate=learning_rate,
            weight_decay=1e-4,
            batch_size=256,
            epochs=EPOCHS[task],
            augmentation=True,
            activation="gelu",
            normalization=normalization,
        )
        elapsed = time.perf_counter() - start
        histories[normalization] = model.training_history
        results.append({
            "normalization": normalization,
            "activation": "gelu",
            "learning_rate": learning_rate,
            "epochs": EPOCHS[task],
            "augmentation": True,
            "validation_accuracy": _validation_accuracy(
                model, dataset, validation_indices, device
            ),
            "final_train_loss": model.training_history[-1]["train_loss"],
            "training_seconds": elapsed,
        })

    _write_csv(
        output_dir / f"{task}_normalization_comparison.csv",
        [
            {"normalization": normalization, **row}
            for normalization, history in histories.items()
            for row in history
        ],
        ("normalization", "epoch", "train_loss"),
    )
    _write_csv(
        output_dir / f"{task}_normalization_validation_results.csv",
        results,
        (
            "normalization",
            "activation",
            "learning_rate",
            "epochs",
            "augmentation",
            "validation_accuracy",
            "final_train_loss",
            "training_seconds",
        ),
    )
    plot_training_curves(
        histories, output_dir / f"{task}_normalization_training_curves.png"
    )
    return results


def run_final_recipe_validation(
    task,
    X_train,
    y_train,
    output_dir="analysis",
    device="cpu",
    seed=0,
    validation_fraction=0.1,
    compare_augmentation=False,
):
    """Validate a task's selected final recipe using training data only."""
    if task not in ("mnist", "cifar10", "imagenet32"):
        raise ValueError(f"Unknown task {task!r}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(X_train) != len(y_train) or len(y_train) < 2:
        raise ValueError("Training arrays must have matching examples")
    if compare_augmentation and task != "imagenet32":
        raise ValueError("Augmentation comparison is requested for ImageNet32 only")

    import numpy as np
    from .train import FINAL_RECIPES

    permutation = np.random.default_rng(seed).permutation(len(y_train))
    validation_count = max(1, int(len(y_train) * validation_fraction))
    validation_indices = permutation[:validation_count].tolist()
    training_indices = permutation[validation_count:].tolist()
    dataset = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train).long())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recipe = FINAL_RECIPES[task]
    augmentation_options = (
        (False, True) if compare_augmentation else (recipe["augmentation"],)
    )
    histories = {}
    results = []
    for use_augmentation in augmentation_options:
        label = f"{task} augmentation={use_augmentation}"
        start = time.perf_counter()
        model = _train_model(
            task,
            X_train,
            y_train,
            seed=seed,
            device=device,
            train_indices=training_indices,
            recipe="submission",
            learning_rate=recipe["learning_rate"],
            weight_decay=recipe["weight_decay"],
            batch_size=recipe["batch_size"],
            epochs=recipe["epochs"],
            augmentation=use_augmentation,
            activation=recipe["activation"],
            normalization=recipe["normalization"],
        )
        elapsed = time.perf_counter() - start
        histories[label] = model.training_history
        results.append({
            "task": task,
            "augmentation": use_augmentation,
            "activation": recipe["activation"],
            "normalization": recipe["normalization"],
            "learning_rate": recipe["learning_rate"],
            "weight_decay": recipe["weight_decay"],
            "batch_size": recipe["batch_size"],
            "epochs": recipe["epochs"],
            "validation_accuracy": _validation_accuracy(
                model, dataset, validation_indices, device
            ),
            "final_train_loss": model.training_history[-1]["train_loss"],
            "training_seconds": elapsed,
        })

    _write_csv(
        output_dir / f"{task}_final_recipe_training.csv",
        [
            {"augmentation": result["augmentation"], **row}
            for result, history in zip(results, histories.values())
            for row in history
        ],
        ("augmentation", "epoch", "train_loss"),
    )
    _write_csv(
        output_dir / f"{task}_final_recipe_validation_results.csv",
        results,
        (
            "task",
            "augmentation",
            "activation",
            "normalization",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "epochs",
            "validation_accuracy",
            "final_train_loss",
            "training_seconds",
        ),
    )
    plot_training_curves(
        histories, output_dir / f"{task}_final_recipe_training_curves.png"
    )
    return results


def run_imagenet_analysis(
    X_train,
    y_train,
    output_dir="analysis",
    device="cuda",
    seed=0,
    validation_fraction=0.1,
    include_ablations=True,
):
    """Train an analysis copy using a training-only split and save metrics."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(X_train) != len(y_train) or len(y_train) < 2:
        raise ValueError("ImageNet32 training arrays must have matching examples")

    import numpy as np

    permutation = np.random.default_rng(seed).permutation(len(y_train))
    validation_count = max(1, int(len(y_train) * validation_fraction))
    validation_indices = permutation[:validation_count].tolist()
    training_indices = permutation[validation_count:].tolist()
    # Reuse the same small training batch for every dynamics checkpoint.
    fixed_indices = training_indices[:min(128, len(training_indices))]
    X_fixed = X_train[fixed_indices]
    y_fixed = y_train[fixed_indices]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dynamics_rows = []

    def observer(model, epoch, update_ratios):
        dynamics_rows.extend(
            measure_dynamics(model, X_fixed, y_fixed, epoch, update_ratios)
        )

    from .train import FINAL_RECIPES

    model = _train_model(
        "imagenet32",
        X_train,
        y_train,
        seed=seed,
        device=device,
        observer=observer,
        train_indices=training_indices,
        **FINAL_RECIPES["imagenet32"],
    )
    _write_csv(output_dir / "imagenet32_dynamics.csv", dynamics_rows, CSV_FIELDS)
    plot_dynamics(dynamics_rows, output_dir / "imagenet32_dynamics.png")

    if not include_ablations:
        return model, None, []

    full_accuracy, ablation_rows = measure_branch_ablations(
        model,
        X_train,
        y_train,
        validation_indices,
        device,
    )
    _write_csv(
        output_dir / "imagenet32_ablations.csv",
        ablation_rows,
        ("block", "accuracy", "accuracy_drop"),
    )
    plot_ablations(ablation_rows, output_dir / "imagenet32_ablations.png")
    return model, full_accuracy, ablation_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="analysis")
    parser.add_argument(
        "--device",
        default=(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dynamics-only",
        action="store_true",
        help="run ImageNet32 training-dynamics measurements without branch ablations",
    )
    parser.add_argument(
        "--compare-task",
        choices=("mnist", "cifar10", "imagenet32"),
        help="compare SGD with AdamW and cosine decay on a training-only split",
    )
    parser.add_argument(
        "--schedule-task",
        choices=("mnist", "cifar10", "imagenet32"),
        help="compare constant-rate AdamW with AdamW and cosine decay",
    )
    parser.add_argument(
        "--learning-rate-task",
        choices=("mnist", "cifar10", "imagenet32"),
        help="compare AdamW cosine training at 5e-4, 1e-3, and 2e-3",
    )
    parser.add_argument(
        "--weight-decay-task",
        choices=("mnist", "cifar10", "imagenet32"),
        help="compare AdamW cosine training with several weight-decay values",
    )
    parser.add_argument(
        "--batch-size-task",
        choices=("mnist", "cifar10", "imagenet32"),
        help="compare batch sizes 128, 256, and 512",
    )
    parser.add_argument(
        "--training-duration-task",
        choices=("mnist", "cifar10", "imagenet32"),
        help="compare short, current, and longer epoch budgets",
    )
    parser.add_argument(
        "--augmentation-task",
        choices=("mnist", "cifar10", "imagenet32"),
        help="compare training with and without task-specific spatial augmentation",
    )
    parser.add_argument(
        "--activation-task",
        choices=("mnist", "cifar10", "imagenet32"),
        help="compare tanh, ReLU, and GELU with the selected training recipe",
    )
    parser.add_argument(
        "--normalization-task",
        choices=("mnist", "cifar10", "imagenet32"),
        help="compare LayerNorm and BatchNorm with GELU and the selected recipe",
    )
    parser.add_argument(
        "--final-recipe-task",
        choices=("mnist", "cifar10", "imagenet32"),
        help="validate the selected final recipe on a training-only split",
    )
    args = parser.parse_args()

    import numpy as np

    task = (
        args.compare_task
        or args.schedule_task
        or args.learning_rate_task
        or args.weight_decay_task
        or args.batch_size_task
        or args.training_duration_task
        or args.augmentation_task
        or args.activation_task
        or args.normalization_task
        or args.final_recipe_task
        or "imagenet32"
    )
    task_dir = Path(args.data_dir) / task
    X_train = np.load(task_dir / "X_train.npy", mmap_mode="c")
    y_train = np.load(task_dir / "y_train.npy", mmap_mode="c")
    if args.final_recipe_task:
        results = run_final_recipe_validation(
            task,
            X_train,
            y_train,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            compare_augmentation=(task == "imagenet32"),
        )
        for result in results:
            print(
                f"task={result['task']}, augmentation={result['augmentation']}: "
                f"validation accuracy {result['validation_accuracy']:.4f}, "
                f"final train loss {result['final_train_loss']:.4f}, "
                f"training time {result['training_seconds'] / 60:.1f} min"
            )
        return
    if args.normalization_task:
        results = run_normalization_comparison(
            task,
            X_train,
            y_train,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
        )
        for result in results:
            print(
                f"normalization={result['normalization']}: validation accuracy "
                f"{result['validation_accuracy']:.4f}, "
                f"final train loss {result['final_train_loss']:.4f}, "
                f"training time {result['training_seconds'] / 60:.1f} min"
            )
        return

    if args.activation_task:
        results = run_activation_comparison(
            task,
            X_train,
            y_train,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
        )
        for result in results:
            print(
                f"activation={result['activation']}: validation accuracy "
                f"{result['validation_accuracy']:.4f}, "
                f"final train loss {result['final_train_loss']:.4f}, "
                f"training time {result['training_seconds'] / 60:.1f} min"
            )
        return

    if args.augmentation_task:
        results = run_augmentation_comparison(
            task,
            X_train,
            y_train,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
        )
        for result in results:
            print(
                f"augmentation={result['augmentation']}, "
                f"lr={result['learning_rate']:g}, epochs={result['epochs']}: "
                f"validation accuracy {result['validation_accuracy']:.4f}, "
                f"final train loss {result['final_train_loss']:.4f}, "
                f"training time {result['training_seconds'] / 60:.1f} min"
            )
        return

    if args.training_duration_task:
        results = run_training_duration_comparison(
            task,
            X_train,
            y_train,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
        )
        for result in results:
            print(
                f"epochs={result['epochs']}, lr={result['learning_rate']:g}: "
                f"validation accuracy {result['validation_accuracy']:.4f}, "
                f"final train loss {result['final_train_loss']:.4f}, "
                f"training time {result['training_seconds'] / 60:.1f} min"
            )
        return

    if args.batch_size_task:
        results = run_batch_size_comparison(
            task,
            X_train,
            y_train,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
        )
        for result in results:
            print(
                f"batch_size={result['batch_size']}, "
                f"lr={result['learning_rate']:g}: validation accuracy "
                f"{result['validation_accuracy']:.4f}, "
                f"final train loss {result['final_train_loss']:.4f}, "
                f"training time {result['training_seconds'] / 60:.1f} min"
            )
        return

    if args.weight_decay_task:
        results = run_weight_decay_comparison(
            task,
            X_train,
            y_train,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
        )
        for result in results:
            print(
                f"weight_decay={result['weight_decay']:g}, "
                f"lr={result['learning_rate']:g}: validation accuracy "
                f"{result['validation_accuracy']:.4f}, "
                f"final train loss {result['final_train_loss']:.4f}, "
                f"training time {result['training_seconds'] / 60:.1f} min"
            )
        return

    if args.learning_rate_task:
        results = run_learning_rate_comparison(
            task,
            X_train,
            y_train,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
        )
        for result in results:
            print(
                f"lr={result['learning_rate']:g}: validation accuracy "
                f"{result['validation_accuracy']:.4f}, "
                f"final train loss {result['final_train_loss']:.4f}, "
                f"training time {result['training_seconds'] / 60:.1f} min"
            )
        return

    if args.schedule_task:
        results = run_schedule_comparison(
            task,
            X_train,
            y_train,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
        )
        for result in results:
            print(
                f"{result['recipe']}: validation accuracy "
                f"{result['validation_accuracy']:.4f}, "
                f"training time {result['training_seconds'] / 60:.1f} min"
            )
        return

    if args.compare_task:
        results = run_optimizer_comparison(
            task,
            X_train,
            y_train,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
        )
        for result in results:
            print(
                f"{result['recipe']}: validation accuracy "
                f"{result['validation_accuracy']:.4f}, "
                f"training time {result['training_seconds'] / 60:.1f} min"
            )
        return

    model, validation_accuracy, ablations = run_imagenet_analysis(
        X_train,
        y_train,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        include_ablations=not args.dynamics_only,
    )
    if args.dynamics_only:
        print("Saved ImageNet32 training dynamics; skipped residual-branch ablations")
        del model
        return
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Measured {len(ablations)} residual-branch ablations")
    del model


if __name__ == "__main__":
    main()
