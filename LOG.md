# Development log

### Baseline — all tasks — 2026-10-21

**Setup:** Instructor package, unchanged residual MLP and plain SGD baseline,
seed 0. MNIST/CIFAR-10 were CPU harness runs.
**Results recorded in `results.csv`:** MNIST test accuracy 96.55% (0.4 and
0.6 min in two runs); CIFAR-10 test accuracy 48.61% (1.1 and 1.2 min in two
runs). The repeated values are reruns, not separate model changes.
**ImageNet32 baseline reference — 2026-10-25:** Trained the starter residual
MLP with SGD (learning rate 0.05), batch size 256, 10 epochs, and seed 0 on all
1,281,167 training examples using MPS. Training took 736.6 seconds (12.3
minutes); final training loss was 5.3616. One evaluation on the official
50,000-example test set yielded 6.69% accuracy. This is a baseline reference
only; it was not used to select or tune settings. Training loss by epoch and
the summary are saved in
`analysis/imagenet32_baseline_reference_training.csv`,
`analysis/imagenet32_baseline_reference_training.png`, and
`analysis/imagenet32_baseline_reference_results.csv`. The run used the existing
starter model architecture and SGD recipe through the analysis training helper.


### Change 1 — shared training recipe — 2026-10-21

**Change:** Replaced plain SGD with AdamW (learning rate 1e-3, weight decay
1e-4) and cosine learning-rate decay. Kept the baseline architecture, batch
size, and epoch counts constant to isolate the optimizer/schedule change.
Changed image loading to retain uint8 training data in host memory and convert
only each minibatch, avoiding a full-dataset float32 copy for ImageNet32.
Added optional ImageNet training-dynamics and residual-branch ablation
measurement code in `src/residual_mlp_baseline/analysis.py`.
**Why:** The baseline's constant-rate plain SGD is deliberately limited. The
recorded CIFAR-10 baseline is 48.61%, so an adaptive optimizer and decaying
learning rate are a focused first experiment. Batch-wise conversion addresses
the approximately 3.9 GB ImageNet32 training image array without changing its
values or labels.
**Expected:** AdamW with cosine decay should improve validation accuracy and
lower training loss within the same epoch budget, though wall-clock time may
increase. Batch-wise conversion should reduce peak host memory substantially.
**Result:** On MNIST, validation accuracy rose from 97.08% (SGD) to 98.13%
(AdamW + cosine), while time increased from 21.9 to 29.9 seconds. Training
loss fell from 0.6732 to 0.2078 for SGD and from 0.4473 to 0.0191 for AdamW.
On CIFAR-10, validation accuracy rose from 51.12% to 51.70%, while time
increased from 58.4 to 78.5 seconds. Training loss fell from 1.8168 to 0.5045
for SGD and from 1.8375 to 0.1132 for AdamW. The validation gains match the
direction expected, though the CIFAR-10 gain is modest and training takes
longer. The repeatable CIFAR-10 loss curves are saved locally in
`analysis/cifar10_training_curves.png`; the comparison data is in
`analysis/cifar10_optimizer_comparison.csv`.
On ImageNet32, the SGD baseline reached 6.30% validation accuracy with final
training loss 5.3986 in 577.4 seconds (9.6 minutes). AdamW with cosine decay
reached 8.83% with final training loss 4.9435 in 961.5 seconds (16.0 minutes).
That is a 2.53 percentage-point validation gain and about 66% more training
time. The lower loss and higher validation accuracy are promising, but this
comparison changes both optimizer and schedule, so it does not isolate which
change caused the gain; it is also a single-seed result. Validation was
measured only on the held-out 10% of training data, not the test set. Loss
curves, per-epoch losses, and summary metrics are saved in
`analysis/imagenet32_training_curves.png`,
`analysis/imagenet32_optimizer_comparison.csv`, and
`analysis/imagenet32_validation_results.csv`.


### Change 2 — learning-rate schedule — 2026-10-23

**Change:** Compare AdamW at a constant learning rate (1e-3) with AdamW and
cosine decay (1e-3 to 0), holding weight decay (1e-4), model architecture,
batch size (256), epoch count, seed, and data split fixed.
**Why:** The first experiment changed optimizer and schedule together. This
follow-up isolates the schedule effect while retaining AdamW in both runs.
**Expected:** Cosine decay may improve validation accuracy late in training
and lower final training loss as the learning rate decreases. It may instead
make little difference within the existing epoch budgets; the comparison will
determine whether the schedule is worth retaining.
**Result:** On MNIST, constant-rate AdamW reached 96.43% validation accuracy
in 31.0 seconds, while cosine-decay AdamW reached 98.13% in 30.0 seconds.
Final training loss was 0.0597 with the constant rate and 0.0191 with cosine
decay. In the recorded CIFAR-10 run, constant-rate AdamW reached 50.66%
validation accuracy in 77.0 seconds, while cosine-decay AdamW reached 51.82%
in 76.9 seconds. Final training loss was 0.2126 with the constant rate and
0.1126 with cosine decay. Cosine decay improved validation accuracy by 1.70
percentage points on MNIST and 1.16 points on CIFAR-10, while training time
was similar. The lower final training loss under cosine decay on both tasks
is consistent with the decreasing learning rate helping optimization late in
training. This supports keeping cosine decay for the current recipe, though
the comparison uses one seed and does not establish that the result
generalizes across seeds. The loss curves and
per-epoch values are in `analysis/mnist_schedule_training_curves.png`,
`analysis/mnist_schedule_comparison.csv`,
`analysis/cifar10_schedule_training_curves.png`, and
`analysis/cifar10_schedule_comparison.csv`; summary metrics are in the matching
`*_schedule_validation_results.csv` files.
On ImageNet32, constant-rate AdamW reached 8.63% validation accuracy in
1,018.9 seconds (17.0 minutes), with final training loss 5.0694. AdamW with
cosine decay reached 8.83% in 1,008.8 seconds (16.8 minutes), with final
training loss 4.9435. Cosine decay improved validation accuracy by 0.20
percentage points and reduced final loss by 0.1259, with nearly identical
runtime. This is a modest single-seed gain, so the evidence for cosine decay
on ImageNet32 is weaker than the gains observed on MNIST and CIFAR-10; repeat
with another seed before treating the result as reliable. The validation split
was taken only from training data; no ImageNet32 test data was used in this
comparison. ImageNet32 curves, per-epoch losses, and summary metrics are in
`analysis/imagenet32_schedule_training_curves.png`,
`analysis/imagenet32_schedule_comparison.csv`, and
`analysis/imagenet32_schedule_validation_results.csv`. 


### Change 3 — learning rate — 2026-10-24

**Change:** Compare AdamW with cosine decay using learning rates 5e-4, 1e-3,
and 2e-3. Keep weight decay (1e-4), architecture, batch size (256), cosine
schedule, epochs, seed, and the training/validation split fixed.
**Why:** The schedule comparison supports cosine decay on MNIST and CIFAR-10.
The next controlled change is to test whether the starting learning rate is
too low, near its best value, or too high.
**Expected:** The current 1e-3 rate is a useful reference. A smaller rate may
underfit within the fixed epoch budget; a larger rate may converge faster but
could become unstable or generalize worse. The validation result, not training
loss alone, will determine the preferred rate.
**Result:** On MNIST, validation accuracies for 5e-4, 1e-3, and 2e-3 were
97.98%, 98.13%, and 97.58%, respectively. Final training losses were 0.0134,
0.0191, and 0.0320; training times were 32.9, 32.1, and 31.2 seconds. On
CIFAR-10, validation accuracies were 52.38%, 51.70%, and 50.22%, respectively.
Final training losses were 0.1222, 0.1132, and 0.0760; training times were
81.6, 83.7, and 82.1 seconds. The 1e-3 rate performed best on MNIST by a small
margin, while 5e-4 performed best on CIFAR-10. The 2e-3 rate had the lowest
training loss on CIFAR-10 but the worst validation accuracy, suggesting
overfitting within this setup. There is no single learning rate that won on
both tasks, supporting task-specific tuning. These results use one seed and
should be treated as an initial comparison rather than a conclusive ranking.
Per-epoch curves and values are saved in the local `analysis/` folder as
`mnist_learning_rate_training_curves.png`,
`mnist_learning_rate_comparison.csv`,
`cifar10_learning_rate_training_curves.png`, and
`cifar10_learning_rate_comparison.csv`; summary results are in each matching
`*_learning_rate_validation_results.csv`.
On ImageNet32, validation accuracy for 5e-4, 1e-3, and 2e-3 was 8.59%, 8.83%,
and 8.59%, respectively. Final training losses were 5.0439, 4.9435, and
4.9827; training times were 20.3, 20.1, and 21.0 minutes. The 1e-3 rate was
best by about 0.24 percentage points, with marginally lower loss and similar
runtime. The two outer rates were effectively tied on validation accuracy;
this single-seed comparison does not establish a robust ranking. As with the
other tasks, the data was split from training data and no test examples were
used. ImageNet32 curves and results are in
`analysis/imagenet32_learning_rate_training_curves.png`,
`analysis/imagenet32_learning_rate_comparison.csv`, and
`analysis/imagenet32_learning_rate_validation_results.csv`.


### Change 4 — weight decay — 2026-10-24

**Change:** Compare AdamW weight decay values 0, 1e-5, 1e-4, and 1e-3 with
cosine decay. Use the best learning rate from Change 3 for each task (1e-3 for
MNIST and 5e-4 for CIFAR-10); keep architecture, batch size, epochs, seed, and
each task's deterministic 90/10 training-only split unchanged.
**Why:** The learning-rate experiment showed that the best rate differed by
task. This experiment tests whether regularization strength also changes the
validation outcome while holding that selected rate fixed.
**Expected:** A modest nonzero weight decay may improve validation accuracy
by reducing overfitting. Zero decay may fit the training set better but
generalize worse; excessive decay may reduce both training and validation
performance. The validation results will determine the setting to carry
forward.
**Result:** On MNIST, validation accuracy for weight decay 0, 1e-5, 1e-4,
and 1e-3 was 98.12%, 98.12%, 98.13%, and 98.15%, respectively. Final training
loss was approximately 0.0191 for all four runs; runtime ranged from 31.0 to
33.0 seconds. On CIFAR-10, validation accuracy was 52.38%, 52.38%, 52.38%,
and 52.36%, respectively. Final training loss was 0.1220, 0.1220, 0.1222,
and 0.1232; runtime ranged from 81.6 to 83.7 seconds. Thus, weight decay had
almost no effect at these values in this one-seed comparison: 1e-3 produced
the highest MNIST score by only 0.03 percentage points, while CIFAR-10 scores
were effectively tied and 1e-3 was marginally lower. These differences are too
small to establish a meaningful best setting without additional seeds. The
loss curves and CSV summaries are in the local `analysis/` folder under the
`mnist_weight_decay_*` and `cifar10_weight_decay_*` filenames.
On ImageNet32, with learning rate 1e-3, validation accuracy for weight decay
0, 1e-5, 1e-4, and 1e-3 was 8.83%, 8.83%, 8.83%, and 8.84%, respectively.
Final training losses were 4.9434, 4.9434, 4.9435, and 4.9437. Runtime ranged
from 12.7 to 21.0 minutes; the slowest two runs were the first executions,
which likely include device warm-up and timing variability, so runtime
differences should not be interpreted as a weight-decay effect. Validation
accuracy was effectively tied across all four settings; the apparent 0.01
percentage-point lead for 1e-3 is negligible at this validation-set size. This
single-seed comparison gives no persuasive reason to change the 1e-4 default.
No test examples were used. ImageNet32 curves and results are saved in
`analysis/imagenet32_weight_decay_training_curves.png`,
`analysis/imagenet32_weight_decay_comparison.csv`, and
`analysis/imagenet32_weight_decay_validation_results.csv`.

### Change 5 — batch size — 2026-10-24

**Change:** Compare mini-batch sizes 128, 256, and 512. Keep AdamW with
cosine decay, weight decay 1e-4, architecture, epochs, seed, and each task's
deterministic 90/10 training-only split fixed. Use the current selected rates:
1e-3 for MNIST and 5e-4 for CIFAR-10.
**Why:** Batch size can affect gradient noise, number of updates per epoch,
throughput, memory consumption, and generalization. The current default is
256; testing one size below and above it gives an initial comparison.
**Expected:** A smaller batch may improve generalization through noisier
updates but take longer. A larger batch may improve throughput but could
generalize worse at a fixed learning rate. Since the epoch count is fixed,
each batch size also results in a different number of optimizer steps; record
this as part of interpreting the experiment.
**Result:** On MNIST (learning rate 1e-3), batch sizes 128, 256, and 512
achieved validation accuracies 98.03%, 98.13%, and 97.72%; final training
losses were 0.0187, 0.0191, and 0.0217; runtimes were 46.2, 30.7, and
21.8 seconds. On CIFAR-10 (learning rate 5e-4), the accuracies were 52.24%,
52.38%, and 52.38%; final losses were 0.1173, 0.1222, and 0.1514; runtimes
were 120.8, 79.5, and 57.5 seconds. Larger batches reduced runtime, but batch
512 did not improve validation accuracy and had higher final training loss.
Batch 256 matched or slightly exceeded the best validation accuracy on both
tasks while taking substantially less time than batch 128. For CIFAR-10,
batch 256 and 512 tied at the reported precision, so this single-seed run does
not show a clear accuracy gain from changing the default. Because epochs were
fixed, the larger batches also meant fewer optimizer updates. The run held
learning rates fixed rather than scaling them with batch size, so the result
compares batch size at the selected per-task rates. Curves and per-epoch CSVs
are in the local `analysis/` folder as `mnist_batch_size_*` and
`cifar10_batch_size_*`; summary metrics are in each matching
`*_batch_size_validation_results.csv`.
On ImageNet32, with learning rate 1e-3, batch sizes 128, 256, and 512 achieved
validation accuracies of 8.80%, 8.83%, and 8.77%. Final training losses were
4.9469, 4.9435, and 4.9792; runtimes were 26.2, 13.6, and 10.8 minutes.
Batch 256 narrowly led in validation accuracy and training loss; batch 512
finished fastest but was 0.06 percentage points lower in accuracy, while
batch 128 was 0.02 points lower and took almost twice as long as batch 256.
These small accuracy differences do not establish a meaningful winner from
one seed. The fixed epoch budget means smaller batches performed more
optimizer updates, and the learning rate was not scaled with batch size.
Batch 256 remains a reasonable speed/accuracy compromise. No test data was
used. ImageNet32 curves and results are saved in
`analysis/imagenet32_batch_size_training_curves.png`,
`analysis/imagenet32_batch_size_comparison.csv`, and
`analysis/imagenet32_batch_size_validation_results.csv`.


### Change 6 — training duration — 2026-10-25

**Change:** Compare 5, 10, and 20 epochs for MNIST and 15, 30, and 45 epochs
for CIFAR-10. Keep AdamW, cosine decay, task-selected learning rate, weight
decay 1e-4, batch size 256, model, seed, and deterministic 90/10
training-only split fixed. For each budget, cosine decay spans that run's
entire epoch budget.
**Why:** The current epoch counts may stop too early or spend time after
validation performance has plateaued. Recording validation accuracy and
training loss each epoch will show whether extra training helps or starts to
overfit.
**Expected:** MNIST may plateau around its current 10 epochs; CIFAR-10 may
benefit from additional epochs, though validation accuracy could level off
before training loss does. Longer runs will take proportionally more time.
**Result:** On MNIST, 5, 10, and 20 epochs reached validation accuracy 97.45%,
98.08%, and 98.28%, with final training losses 0.0588, 0.0183, and 0.0021.
Training times were 17.6, 32.1, and 63.2 seconds. Validation accuracy still
improved at 20 epochs, although the gain over 10 epochs was small (+0.20
percentage points) for nearly twice the training time. On CIFAR-10, 15, 30,
and 45 epochs reached 54.50%, 52.50%, and 52.36% validation accuracy, with
final losses 0.7541, 0.1279, and 0.0024. Training times were 42.6, 84.2, and
123.4 seconds. CIFAR-10 validation accuracy peaked at 15 epochs while training
loss continued to fall substantially in the longer runs; this is consistent
with overfitting, though the cosine schedule's horizon also differs across
budgets and may affect the comparison. The epoch-by-epoch curves and values
are saved locally as `analysis/mnist_training_duration.png`,
`analysis/mnist_training_duration_by_epoch.csv`,
`analysis/cifar10_training_duration.png`, and
`analysis/cifar10_training_duration_by_epoch.csv`; final-budget summaries are
in the matching `*_training_duration_results.csv` files.
On ImageNet32, the 5-, 10-, and 20-epoch runs reached endpoint validation
accuracies of 7.41%, 8.83%, and 8.65%, with final training losses 5.3409,
4.9435, and 4.3481. Training times were 7.7, 16.0, and 33.2 minutes.
Although the 20-epoch run had the lowest training loss, its endpoint
validation accuracy was 0.18 percentage points below the 10-epoch run. In
the 20-epoch run, validation accuracy peaked at epoch 13 (9.02%) and then
declined to 8.65% by epoch 20 while training loss continued to fall, a pattern
consistent with overfitting. However, each budget used its own cosine-schedule
horizon, so the endpoint comparison also changes the schedule; the epoch-13
peak is within the 20-epoch trajectory and does not make the separate 10-epoch
run a matched checkpoint. Treat 10 epochs as the current validation candidate,
not a conclusive optimum, and verify with another seed or a fixed-schedule
checkpoint comparison. No test data was used. ImageNet32 epochwise metrics,
summary results, and plot are saved in
`analysis/imagenet32_training_duration_by_epoch.csv`,
`analysis/imagenet32_training_duration_results.csv`, and
`analysis/imagenet32_training_duration.png`.


### Change 7 — data augmentation — 2026-10-25

**Change:** Compare training without augmentation against task-specific spatial
augmentation: random translations up to 2 pixels for MNIST; reflected random
crops up to 4 pixels and random horizontal flips for CIFAR-10. Keep the
architecture, optimizer, cosine schedule, selected learning rates, weight
decay, batch size, epochs, seed, and deterministic 90/10 training-only split
fixed.
**Why:** Small spatial variations may make the model less sensitive to image
position and improve generalization without adding outside data.
**Expected:** Augmentation may improve validation accuracy, especially for
CIFAR-10, though it may also slow training or make fitting harder. MNIST is
already highly accurate, so its improvement may be smaller.
**Result:** On MNIST, validation accuracy was 97.97% without augmentation and
97.98% with translations. Final training losses were 0.0210 and 0.0716, and
runtime was 33.1 and 33.0 seconds. The accuracy difference is
negligible, while augmentation made the training objective harder. On CIFAR-10,
validation accuracy was 51.26% without augmentation and 56.94% with random
crops and horizontal flips, a +5.68 percentage-point gain. Final training
losses were 0.1071 and 1.1965; runtime was 81.1 and 85.4 seconds. The
augmented model's much higher training loss is expected because each minibatch
contains randomly transformed images, while validation uses the original
images. In this controlled paired run, augmentation had almost no effect on
MNIST and substantially improved CIFAR-10 validation accuracy. The
unaugmented CIFAR-10 score differs from earlier experiments because this
comparison now uses a dedicated DataLoader shuffle generator to ensure
augmentation does not alter the example order between the paired runs. These
are single-seed results and should be repeated before finalizing. Curves and
CSV results are in the local `analysis/` folder as
`mnist_augmentation_training_curves.png`,
`mnist_augmentation_comparison.csv`,
`cifar10_augmentation_training_curves.png`, and
`cifar10_augmentation_comparison.csv`; summary metrics are in the matching
`*_augmentation_validation_results.csv` files.
On ImageNet32, validation accuracy was 8.83% without augmentation and 8.25%
with random padded crops and horizontal flips, a 0.57 percentage-point
decrease. Final training losses were 4.9435 and 5.3706, respectively, and
training times were 15.7 and 19.7 minutes. The augmented training loss is
measured on transformed examples, so it is not directly comparable to
unaugmented training loss. In this single-seed run, augmentation made
optimization harder, increased runtime by about 25%, and did not improve
validation accuracy. Keep augmentation off for ImageNet32 unless another
seed produces evidence otherwise. Validation used only held-out training
examples; no test data was used. ImageNet32 loss curves and results are in
`analysis/imagenet32_augmentation_training_curves.png`,
`analysis/imagenet32_augmentation_comparison.csv`, and
`analysis/imagenet32_augmentation_validation_results.csv`.



### Change 8 — activation — 2026-10-25

**Change:** Compare tanh, ReLU, and GELU while keeping each task's model
dimensions, normalization, residual connections, initialization, optimizer,
cosine schedule, learning rate, weight decay, batch size, epochs, augmentation,
seed, and training-only validation split fixed.
**Why:** The model currently uses tanh throughout. Activation functions change
the nonlinear response and gradient flow, which may matter for the 30-block
residual MLP and for the different task difficulties.
**Expected:** ReLU or GELU may optimize differently from tanh; the winner may
differ between MNIST and CIFAR-10. Validation accuracy, not training loss
alone, will decide whether to change the default.
**Result:** On MNIST, tanh, ReLU, and GELU achieved validation accuracies of
97.98%, 98.52%, and 98.62%; final training losses were 0.0716, 0.0472, and
0.0422; runtimes were 32.7, 28.5, and 32.1 seconds. On CIFAR-10, validation
accuracies were 56.94%, 60.14%, and 60.50%; final training losses were 1.1965,
1.0538, and 1.0166; runtimes were 87.8, 77.9, and 88.7 seconds. GELU was best
on both tasks in this single-seed comparison, with ReLU close behind; both
improved over tanh. Training loss also fell for ReLU/GELU. CIFAR-10 augmented
training loss remains much higher than validation-style unaugmented loss
because each training batch is transformed. These are validation results only
and require confirmation with another seed before changing the final recipe.
Curves and CSV results are in the local `analysis/` folder as
`mnist_activation_training_curves.png`,
`mnist_activation_comparison.csv`,
`cifar10_activation_training_curves.png`, and
`cifar10_activation_comparison.csv`; summary results are in the matching
`*_activation_validation_results.csv` files.
On ImageNet32, tanh,
ReLU, and GELU reached validation accuracies of 8.25%, 9.84%, and 11.07%;
final training losses were 5.3706, 5.1406, and 4.9960, and runtimes were
15.3, 20.9, and 22.2 minutes. GELU was the strongest in this seed-0 run,
improving validation accuracy by 2.82 percentage points over tanh and 1.23
points over ReLU. It also had the lowest training loss, though its training
time was the highest. Augmentation was enabled for all variants, so this
result compares activations within the augmented recipe; the prior
augmentation comparison found that recipe worse than no augmentation on
ImageNet32. This is a single-seed result and should be confirmed before
finalizing. No test data was used. ImageNet32 curves and results are in
`analysis/imagenet32_activation_training_curves.png`,
`analysis/imagenet32_activation_comparison.csv`, and
`analysis/imagenet32_activation_validation_results.csv`.


### Change 9 — normalization — 2026-10-25

**Change:** Compare LayerNorm with BatchNorm in every residual block and the
classification head. Hold GELU, initialization, AdamW with cosine decay,
selected task learning rates, weight decay, batch size, epochs, augmentation,
seed, and deterministic 90/10 training-only split fixed.
**Why:** The baseline uses LayerNorm throughout. BatchNorm is another allowed
normalization and uses minibatch activation statistics during training; the
comparison tests whether it helps this residual MLP.
**Expected:** BatchNorm may improve optimization or validation accuracy, but
its running statistics and dependence on minibatch composition could also make
it less suitable than LayerNorm. The effect may differ between datasets.
**Result:** BatchNorm was slightly better on the fixed seed-0 validation split:

| Dataset | Normalization | Validation accuracy | Final train loss | Training time |
|---|---|---:|---:|---:|
| MNIST | LayerNorm | 98.62% | 0.0422 | 33.4 s |
| MNIST | BatchNorm | 98.67% | 0.0421 | 36.2 s |
| CIFAR-10 | LayerNorm | 60.50% | 1.0166 | 87.4 s |
| CIFAR-10 | BatchNorm | 61.28% | 1.0456 | 97.3 s |

BatchNorm gained 0.05 percentage points on MNIST and 0.78 points on CIFAR-10,
while taking about 8% and 11% longer, respectively. CIFAR-10's BatchNorm run
had a higher final training loss despite better validation accuracy. These
single-seed results are evidence for a possible validation benefit, not proof
that BatchNorm generalizes better; repeat with additional seeds before changing
the default.
On ImageNet32, LayerNorm reached 11.07% validation accuracy with final
training loss 4.9960 in 14.6 minutes; BatchNorm reached 10.84% with loss
5.0316 in 18.0 minutes. LayerNorm was ahead by 0.23 percentage points and
took about 3.4 minutes less. This suggests LayerNorm is the better choice for
the tested ImageNet32 recipe, but the difference is small and based on one
seed. Augmentation and GELU were used for both runs, so these results are
conditional on that recipe. No test data was used. ImageNet32 results and
curves are saved in `analysis/imagenet32_normalization_validation_results.csv`,
`analysis/imagenet32_normalization_comparison.csv`, and
`analysis/imagenet32_normalization_training_curves.png`.
**Artifacts:** `analysis/mnist_normalization_validation_results.csv`,
`analysis/cifar10_normalization_validation_results.csv`, the corresponding
`*_normalization_comparison.csv` loss histories, and
`*_normalization_training_curves.png`.

### ImageNet32 training-dynamics analysis — 2026-10-26

**Setup:** Seed 0, MPS, 90/10 training-only split, and the selected final
ImageNet32 recipe: 30-block residual MLP (width 128, hidden width 256), GELU,
LayerNorm, AdamW (learning rate 1e-3, weight decay 1e-4), cosine decay, batch
size 256, 10 epochs, and random padded crops/horizontal flips during training.
Reused the same fixed batch of 128 unaugmented training examples at
initialization and after each epoch, with the model in evaluation mode.
Gradients are measured on that fixed batch; update-to-weight ratios are
measured from the final optimizer step of each epoch. The analysis records 30
block measurements and 63 linear-layer measurements at each of 11 checkpoints
(1,023 rows total). No test data was used. Per request, this was dynamics-only;
no residual-branch ablations were run.
**Results:** Training loss fell from 6.0973 after epoch 1 to 4.9960 after
epoch 10. The table gives means across all residual blocks or linear weights:

| Checkpoint | Block activation variance | Branch/stream RMS | Linear weight gradient RMS | Update/weight RMS |
|---|---:|---:|---:|---:|
| Initialization | 4.181 | 0.305 | 7.72e-4 | n/a |
| Epoch 1 | 169.517 | 0.060 | 2.57e-4 | 3.87e-3 |
| Epoch 5 | 202.404 | 0.074 | 4.23e-4 | 1.58e-3 |
| Epoch 10 | 166.040 | 0.086 | 5.86e-4 | 5.32e-5 |

GELU is used in this recipe, so tanh saturation is not applicable.
**Interpretation:** Activation variance increases steeply with depth at
initialization (0.820 in block 0 to 7.339 in block 29). Training raises it
substantially: after epoch 1 it ranges from 141.307 to 177.409, peaks around
epoch 5, and remains high at epoch 10 (151.837–210.636). LayerNorm controls
the normalized inputs to each branch, but the residual stream itself still
accumulates large variance. The branch/stream RMS ratio falls through depth
and is much smaller after training; by epoch 10 it declines from 0.289 in the
first block to 0.055 in the final block. Thus later residual branches are
smaller relative to their input stream, although none of these measurements
alone establishes whether a block is important to prediction. Fixed-batch
gradient RMS does not vanish monotonically across the residual blocks; after
epoch 10 it ranges from 2.42e-4 to 1.11e-3 across residual linear weights,
while the head's first linear layer is 3.50e-3. The average update-to-weight
ratio drops from 3.87e-3 at epoch 1 to 5.32e-5 at epoch 10, consistent with
the cosine schedule's smaller late-training steps. Measurements are from one
fixed batch and one seed, so they describe this run rather than the full
training distribution.
**Artifacts:** `analysis/imagenet32_dynamics.csv` contains per-block and
per-linear-layer values at every checkpoint. The figure
`analysis/imagenet32_dynamics.png` plots initialization, epoch 1, epoch 5, and
epoch 10 for readability; activation variance uses a logarithmic axis, and
tanh saturation is marked not applicable for this GELU model. The analysis
code and dynamics-only command are in
`src/residual_mlp_baseline/analysis.py`; use
`PYTHONPATH=src python3 -m residual_mlp_baseline.analysis --device mps --dynamics-only`
to reproduce without running ablations.

### Final recipe validation and evaluation — 2026-10-26

**Selected recipes:** Implemented in `FINAL_RECIPES` in
`src/residual_mlp_baseline/train.py`. MNIST uses GELU, BatchNorm, AdamW
(learning rate 1e-3, weight decay 1e-4), cosine decay, batch size 256, 10
epochs, and no augmentation. CIFAR-10 uses GELU, BatchNorm, AdamW (learning
rate 5e-4, weight decay 1e-4), cosine decay, batch size 256, 15 epochs, and
random reflected crops up to 4 pixels plus horizontal flips. ImageNet32 uses
GELU, LayerNorm, AdamW (learning rate 1e-3, weight decay 1e-4), cosine decay,
batch size 256, 10 epochs, and the same color-image augmentation.
**Training-only validation:** Seed 0, deterministic 90/10 split, no test
examples. MNIST reached 98.50% validation accuracy (final training loss
0.0036; 37.7 seconds); CIFAR-10 reached 58.02% (loss 1.1973; 49.6 seconds).
For ImageNet32, the selected GELU/LayerNorm recipe achieved 10.10% without
augmentation and 11.07% with augmentation. Losses were 4.2963 and 4.9960;
times were 17.9 and 16.1 minutes. Although augmented training loss is
measured on transformed examples, validation uses the unaugmented fixed split.
The 0.97 percentage-point validation advantage led to selecting augmentation
for ImageNet32. This choice is based on one seed. Results are saved in
`analysis/mnist_final_recipe_validation_results.csv`,
`analysis/cifar10_final_recipe_validation_results.csv`, and
`analysis/imagenet32_final_recipe_validation_results.csv`, with training
histories in the corresponding `*_final_recipe_training.csv` files.
**Full-data evaluation:** After selecting the recipes, called the package
`train()` on each complete training set and `predict()` once on its official
test set (seed 0; CPU for MNIST/CIFAR-10, MPS for ImageNet32). MNIST reached
98.54% test accuracy in 40.6 seconds, compared with the recorded baseline
96.55%. CIFAR-10 reached 58.00% in 53.9 seconds, compared with 48.61%.
ImageNet32 reached 10.26% in 1,201.8 seconds (20.0 minutes), compared with
the starter baseline's 6.69%. The test scores were recorded after recipe
selection and were not used for further tuning. Prediction shapes were
`(10000, 10)`, `(10000, 10)`, and `(50000, 1000)`; maximum absolute row-sum
errors from 1.0 were 2.38e-7, 2.38e-7, and 3.58e-7, respectively. Full
results are in `analysis/final_submission_evaluation.csv`; individual final
loss histories are in `analysis/*_final_submission_training.csv`.
This was a local API evaluation. The course harness was subsequently rerun
from a restarted kernel using the GitHub package; its end-to-end results are
recorded below and in `results.csv`.

### Restarted-kernel course harness verification — 2026-10-26

Restarted the notebook kernel, reinstalled the package from
`git+https://github.com/dauntless15/AI4200-imagenet32x32-baseline-model.git`,
and ran all three tasks with seed 0. The MNIST, CIFAR-10, and ImageNet32
training runs completed for 10, 15, and 10 epochs, respectively. The harness
reported 98.54% MNIST accuracy in 0.7 minutes, 58.00% CIFAR-10 accuracy in
0.9 minutes, and 10.23% ImageNet32 accuracy in 14.7 minutes. Its shape and
probability-row-sum assertions passed for every task; the results were written
to `results.csv`. These are final test-set evaluations after recipe selection,
not inputs to further tuning. Timings differ from earlier local API tests and
should be understood as environment-specific.
