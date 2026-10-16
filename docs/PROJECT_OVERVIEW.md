# Project overview

## Question and motivation

Can a computer control a real-time mobile game using only the phone screen
and touch commands? Playing Honor of Kings motivated this project: decisions
that feel intuitive to a player become concrete engineering problems when
they must be represented as visual evidence, actions, and delayed feedback.

The system connects an iQOO 10 (Android 16) to a Jetson AGX Orin running Linux.
It captures the scrcpy window, encodes images with a frozen ResNet-101, predicts
actions with a Transformer, and sends touch commands back to the phone.
Manual takeover and trajectory recording support further data collection.

## Individual project work

The local work includes adapting capture and touch control to Linux/Jetson and
the Android phone, integrating collection and manual control, investigating
state-recognition failures, and developing a hybrid visual/OCR reward pipeline.
It also includes grouped classifier validation, targeted data collection,
per-image review tools, and integration with offline policy training.
AI coding assistance was used in implementation, debugging, and documentation;
the accompanying portfolio should describe the applicant's own decisions and
experiments rather than imply all source lines were written unaided.

## What changed during the investigation

The original six-class visual approach struggled with short-lived kill icons.
Instead of treating every event as an image-classification problem, the revised
system uses three visual classes and temporal OCR of HUD counters. Gold, kill,
and death changes provide inspectable event evidence. Gold is a balance rather
than cumulative earnings, so purchases and OCR errors require explicit checks.

Tower targeting remains a visual problem. Error inspection led to collecting
more tower-attack examples, including difficult directions and tower types,
and retraining the ResNet-50 classifier at 384 x 192 input resolution.

## Recorded evidence

The curated policy dataset used in the documented experiment contains seven
recordings and 19,634 frames. The applicant reports collecting roughly ten games
overall; recorded segments, complete games, and independent examples are not
interchangeable. Neighboring frames often share almost the same information.

For the visual classifier, the added-data experiment used 19,175 training images
(including augmentations) and 5,125 original validation images separated by run.
Its best checkpoint was epoch 14, selected by validation macro F1.

On 2,054 three-class images from recording `1787743636`, the saved evaluation
reported 2,050 correct predictions (99.81% accuracy). For tower attacks, 32 of
35 positives were detected, with one false positive: precision 96.97%, recall
91.43%, and F1 94.12%.

These are classifier results, not gameplay win rates. The recording was held
out of classifier fitting, but its errors had been inspected during development;
the repeated evaluation is not a blind test of the full development process.
The same recording was later included in policy training, so it is not a held-out
test for that policy. A new game is needed for a fresh generalization assessment.

The applicant reports that the running agent can leave the base, follow the
lane, engage in basic laning, and has defeated a lowest-level Bronze opponent.
These are observed milestones rather than a controlled, repeated win-rate study.

## Current scope and remaining work

The runtime captures, infers, acts, and saves trajectories. Offline training
weights recorded action log probabilities using inferred rewards. Standard
on-policy PPO, automated episode resets, valid terminal flags, and an automatic
collect/update loop have not been completed.

The current action-history embedding is multiplied by zero, and the value
head is not trained by a value loss in the current policy workflow. Signed
reward weighting also has an unbounded objective on negative samples. These
limitations are retained and disclosed rather than silently changing the
algorithm during repository cleanup.

Next experiments should measure repeated gameplay outcomes, takeover events,
and per-stage latency; collect independent scenes; and compare a properly
implemented rollout-based PPO update with the offline baseline.

## Evidence availability

Source code and tests are included. This local checkout also has the runtime
models in a Git-ignored `weights/` directory; source-only clones need them
supplied separately. Raw trajectories, plots, screenshots, and full evaluation
reports are not bundled in this repository.
The figures above summarize the existing local experiment artifacts; this code
checkout alone is insufficient to reproduce the exact historical metrics.
README commands explain how to migrate supplied artifacts and regenerate review
reports. The application portfolio can accompany this repository with its
screenshots, representative errors, and experiment tables.
