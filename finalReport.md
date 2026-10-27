# AI 4200 Project 1 — Final Report

> **Status:** This is the requested Markdown report. The final training and
> test evaluation used the repository's `train()` and `predict()` API, but the
> course notebook has not yet been rerun from a fresh environment. Export this
> report to PDF and complete that fresh-environment harness check before
> submission.

## Final models and results

All three models use the same residual MLP architecture: flatten the image,
project to width 128, pass through 30 residual blocks with hidden width 256,
then use a two-linear-layer classification head. Each block normalizes the
128-dimensional stream, expands it from 128 to 256, applies GELU, projects
back to 128, and adds the skip connection. The head normalizes, expands to
256, applies GELU, and projects to the task's class count. Linear layers use
Xavier-normal initialization with zero biases. Inputs are scaled to [-1, 1].
Each model was trained from scratch with cross-entropy and only its task's
provided training data.

| Task | Activation / normalization | Optimizer and schedule | Batch / epochs | Augmentation | Baseline test accuracy | Final test accuracy | Change | Training time |
|---|---|---|---|---|---:|---:|---:|---:|
| MNIST | GELU / BatchNorm | AdamW, lr 0.001, weight decay 0.0001, cosine | 256 / 10 | None | 96.55% | 98.54% | +1.99 pp | 40.6 s |
| CIFAR-10 | GELU / BatchNorm | AdamW, lr 0.0005, weight decay 0.0001, cosine | 256 / 15 | Reflect-padded random crops up to 4 pixels; horizontal flips | 48.61% | 58.00% | +9.39 pp | 53.9 s |
| ImageNet32 | GELU / LayerNorm | AdamW, lr 0.001, weight decay 0.0001, cosine | 256 / 10 | Reflect-padded random crops up to 4 pixels; horizontal flips | 6.69% | 10.26% | +3.57 pp | 20.0 min |

Baseline MNIST and CIFAR-10 scores and times are from the existing CPU harness
records in `results.csv`; ImageNet32 baseline is the seed-0 MPS reference in
`LOG.md`. Final results were measured with seed 0 by calling the package
`train()` and `predict()` functions on the full training and test arrays,
respectively. Final test data was evaluated only after recipe selection and
was not used for tuning. Training time measures training, not inference.

The final predictions had the expected shapes: `(10000, 10)` for MNIST,
`(10000, 10)` for CIFAR-10, and `(50000, 1000)` for ImageNet32. The maximum
absolute row-sum errors from 1.0 were at most `3.58e-7`.

The final recipe settings are implemented in `src/residual_mlp_baseline/train.py`.
The required public interfaces remain:

```python
model = train(task, X_train, y_train, seed=0, device="cuda")
probs = predict(model, X_test)
```

## Experiments that most informed the recipe

Results below are single-seed, training-only validation comparisons unless
explicitly identified as final test results. Full settings and caveats are in
`LOG.md`.

1. **CIFAR-10 augmentation:** In the paired seed-0 comparison, random crops and
   horizontal flips improved validation accuracy from 51.26% to 56.94%
   (+5.68 percentage points). The complete chosen 15-epoch GELU/BatchNorm
   recipe reached 58.02% validation accuracy.
2. **ImageNet32 optimizer and schedule:** The initial comparison of SGD with
   AdamW plus cosine decay improved validation accuracy from 6.30% to 8.83%
   (+2.53 points), at the cost of longer training. Because both optimizer and
   schedule changed together, that run does not isolate their individual
   effects.
3. **GELU activation:** GELU was the strongest activation in the tested
   recipes. On ImageNet32 with augmentation, it reached 11.07% validation
   accuracy versus 9.84% for ReLU and 8.25% for tanh.
4. **Task-specific learning rates:** The validation comparisons favored
   0.001 for MNIST and ImageNet32 and 0.0005 for CIFAR-10, motivating
   task-specific learning rates.
5. **Normalization:** BatchNorm slightly led LayerNorm on the tested MNIST and
   CIFAR-10 recipes; LayerNorm slightly led BatchNorm on ImageNet32. These
   margins were small and are not conclusive from one seed.

## ImageNet32 training dynamics

The selected ImageNet32 recipe was measured at initialization and after each
epoch using the same fixed batch of 128 training examples in evaluation mode.
Across the 30 residual blocks, activation variance grew through depth: at
initialization it rose from 0.82 in block 0 to 7.34 in block 29. At epoch 10,
it ranged from 151.84 to 210.64. LayerNorm stabilizes each branch's normalized
input, but the residual stream itself accumulates large variance.

The branch-to-stream RMS ratio decreased through depth and was lower after
training. It fell from 0.80 in block 0 to 0.17 in block 29 at initialization,
and from 0.29 to 0.055 at epoch 10. Thus later residual branches are smaller
relative to their input stream in this run. This measurement alone does not
establish that those blocks are unimportant.

Fixed-batch gradient RMS did not vanish monotonically across residual-layer
weights: at epoch 10 it ranged from `2.42e-4` to `1.11e-3`. The head's first
linear layer had gradient RMS `3.50e-3`. Mean update-to-weight RMS declined
from `3.87e-3` at epoch 1 to `5.32e-5` at epoch 10, consistent with the
cosine schedule's smaller late-training steps. Saturation is not applicable
because this final model uses GELU rather than tanh. These are measurements of
one fixed batch and one seed, not estimates over the full dataset.

![ImageNet32 training dynamics by depth and checkpoint](analysis/imagenet32_dynamics.png)

The per-block and per-layer values are in `analysis/imagenet32_dynamics.csv`.
This project used dynamics-only analysis; no 5000-level branch ablation was
performed.

## Training curves and approaches that did not help

The figure compares CIFAR-10 training loss for the earlier SGD baseline and
the final selected recipe. The baseline uses 30 epochs without augmentation;
the final recipe uses 15 epochs with augmentation. Because the training
conditions differ and augmentation makes training examples harder, these
curves are useful context rather than a controlled comparison.

![CIFAR-10 baseline and final-recipe training losses](analysis/cifar10_baseline_vs_final_training_curves.png)

Weight decay values from 0 through 0.001 produced essentially tied validation
accuracy on the tested ImageNet32 recipe, so there is no evidence here to tune
away from 0.0001. The ImageNet32 duration experiment showed that training loss
continued to fall at 20 epochs while validation accuracy peaked at 9.02% at
epoch 13 and ended at 8.65%; its separate 10-epoch run ended at 8.83%. Each
duration had its own cosine horizon, so this is not a matched-checkpoint
comparison.

The effect of augmentation on ImageNet32 depended on the rest of the recipe.
An earlier tanh comparison showed a decrease with augmentation, but the
final-recipe GELU/LayerNorm comparison favored augmentation: 11.07% versus
10.10% validation accuracy without it. Accordingly, the final ImageNet32
recipe uses augmentation. This is why the final choice was based on a
training-only comparison using the selected activation and normalization,
rather than applying the earlier isolated result blindly.

## Task-specific choices and interpretation

MNIST and CIFAR-10 use BatchNorm, while ImageNet32 uses LayerNorm. MNIST does
not use augmentation; the color-image tasks use random crops and horizontal
flips. CIFAR-10 uses the lower learning rate and shorter 15-epoch budget.
These differences reflect validation evidence and different dataset
complexities; they do not prove that the same choices generalize across
seeds.

The MLP can learn useful image classifiers, but flattening pixels does not
provide the spatial locality and translation structure built into
convolutional models. Data augmentation supplies some translation and
reflection variation during training, but it does not add a convolutional
inductive bias. ImageNet32's 1,000 classes and large training set also make
the task substantially harder for this compact residual MLP, as reflected by
its lower accuracy.

AdamW applies decoupled weight decay while adaptively scaling parameter
updates. Cosine decay gradually reduces the learning rate over the fixed
training budget. GELU supplies a smooth nonlinear activation. BatchNorm
normalizes features using minibatch statistics during training and running
statistics at inference; LayerNorm normalizes each example's feature vector
independently. Random padded crops and horizontal flips encourage robustness
to small spatial changes without using extra data or additional models.

## Reproducibility and remaining submission checks

- Final recipe validation and augmentation selection are recorded in
  `analysis/mnist_final_recipe_validation_results.csv`,
  `analysis/cifar10_final_recipe_validation_results.csv`, and
  `analysis/imagenet32_final_recipe_validation_results.csv`.
- Final full-training evaluation and output-shape checks are recorded in
  `analysis/final_submission_evaluation.csv`.
- The baseline-vs-final CIFAR-10 loss histories are in
  `analysis/cifar10_baseline_vs_final_training_curves.png`.
- The final harness has **not** been run from a fresh environment, and this
  Markdown report has **not** yet been exported to PDF.
- No 5000-level ablation was run, per the user's instruction.