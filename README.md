[MASTER_README_Airplanes_v2.md](https://github.com/user-attachments/files/25827775/MASTER_README_Airplanes_v2.md)
# Airplanes_v2 — From a 5-Class Prototype to a Scalable FGVC-Aircraft Recognition System

## Project overview

This repository documents the full progression of an aircraft image-classification project, from an initial **small-scale proof of concept** to a substantially more ambitious **fine-grained visual classification pipeline** based on the **FGVC-Aircraft** benchmark.

The project was built in two major phases:

1. **An initial five-class prototype**, created as a fast and practical entry point into deep learning for image classification.
2. **A larger FGVC-Aircraft pipeline**, designed to move from a toy problem to a significantly more demanding real-world computer vision challenge covering:
   - many more aircraft categories,
   - structured preprocessing,
   - reproducible training,
   - test-set evaluation,
   - deployment through Hugging Face Spaces.

The objective of this repository is not only to present a working model, but also to show a clear engineering progression:
- start from a small and controlled baseline,
- validate the end-to-end pipeline,
- identify limitations,
- redesign the training workflow,
- scale the project to a more serious dataset,
- export and deploy the result.

This progression is central to the value of the project. It shows the transition from a tutorial-inspired prototype to a substantially more rigorous machine learning system.

---

## Why this project matters

Aircraft recognition is a particularly interesting computer vision problem because it combines:
- **fine-grained visual differences** between related aircraft types,
- strong variation in viewpoint, lighting, background, and image quality,
- subtle shape cues that can easily be lost with weak preprocessing,
- a natural hierarchy of labels:
  - manufacturer,
  - family,
  - variant.

This makes the problem much more demanding than a simple small-class image classifier.

The repository therefore shows two levels of work:

- a **beginner-friendly initial classifier** on five classes,
- a **more advanced fine-grained classification pipeline** designed around the FGVC-Aircraft dataset and engineering-oriented evaluation logic.

---

## Project trajectory

### Phase 1 — Initial five-class prototype

The project started with a deliberately small-scale classifier trained on **five aircraft classes**.

This first version was inspired by the following tutorial:

- *Build your first deep learning app within an hour*  
  `https://medium.com/data-science/build-your-first-deep-learning-app-within-an-hour-4e80c120e99f`

The point of this first stage was not to solve aircraft recognition at scale.  
It was to validate the essential components of the workflow:

- loading image data,
- building a FastAI classification pipeline,
- training a first convolutional model,
- exporting the artifact,
- serving the model through a lightweight app.

This stage provided a first working deep learning application and a practical understanding of:
- data pipelines,
- transfer learning,
- model export,
- simple deployment workflows.

### Phase 2 — First Hugging Face deployment

Once the first prototype worked locally, the next step was to deploy an image-classification demo on **Hugging Face Spaces**.

This phase introduced several practical engineering questions that are often invisible in toy notebooks:

- how to package a model for deployment,
- how to handle dependencies reproducibly,
- how to structure a minimal inference application,
- how to move from a notebook environment to a stable app,
- how to expose predictions through a clean interface.

This first deployment was useful not because the model was large, but because it forced the transition from **training code** to **usable product-like code**.

### Phase 3 — Scaling the problem with FGVC-Aircraft

After the initial success of the five-class prototype, the main limitation became obvious: the scope was far too narrow.

The project was then redesigned around the **FGVC-Aircraft** dataset, a benchmark for fine-grained aircraft recognition.

This changed the nature of the problem completely.

Instead of classifying a few hand-picked aircraft classes, the project now had to address:
- a much larger label space,
- stronger intra-class similarity,
- higher inter-class ambiguity,
- the need for structured preprocessing,
- the need for proper test-set evaluation.

The project therefore moved from:
- a **simple educational prototype**
to
- a **structured fine-grained visual classification system**.

### Phase 4 — Toward broader coverage

The larger FGVC-based model was designed to move toward:
- **dozens of manufacturers**,
- **a broad set of aircraft families**,
- and a more credible deployment target than the initial five-class demo.

The current repository therefore contains both:
- the **historical first prototype**,
- and the **larger production-oriented training pipeline** that superseded it.

---

## Repository structure

```text
Airplanes_v2/
│
├── .gitattributes
├── runtime.txt
│
├── demo-5-models/
│   ├── airplanes_5_models.ipynb
│   ├── airplanes_5_models.pkl
│   ├── app_5_models.py
│   ├── README.md
│   └── requirements.txt
│
├── Hugging Face demo/
│   ├── .gitattributes
│   ├── app.py
│   ├── Dockerfile
│   ├── labels.json
│   ├── model.pth
│   ├── README.md
│   ├── requirements.txt
│   └── supported_families.json
│
└── training/
    ├── confusion_matrix_row_normalized_test.png
    ├── fgvc-manufacturer-resnet34.ipynb
    ├── fgvc_manufacturer_resnet34_optimized_fastai.pkl
    ├── labels.json
    ├── model.pth
    ├── test_best_classes.csv
    ├── test_metrics_summary.csv
    ├── test_metrics_summary_tta.csv
    ├── test_per_class_report.csv
    ├── test_predictions_top5.csv
    ├── test_top_confusions.csv
    └── test_worst_classes.csv
```

---

## What each folder contains

### `demo-5-models/`

This folder contains the **initial prototype** of the project.

It is the simplest version of the aircraft-recognition pipeline and should be understood as the first proof of concept.

#### Files

- `airplanes_5_models.ipynb`  
  The notebook for the first five-class experiment.  
  It documents the earliest stage of the project: a small-scale classifier used to validate the basic workflow.

- `airplanes_5_models.pkl`  
  The exported FastAI model artifact for the five-class classifier.

- `app_5_models.py`  
  The lightweight inference application associated with the five-class demo.

- `README.md`  
  Documentation specific to the five-class demo.

- `requirements.txt`  
  Python dependencies required for the five-class environment.

#### Role in the project

This folder is important because it shows the **starting point** of the work:
- simpler data,
- simpler objective,
- simpler deployment,
- simpler evaluation.

It demonstrates the first successful version of the project before the transition to the larger FGVC-Aircraft setup.

---

### `training/`

This folder contains the **main training and evaluation artifacts** for the larger FGVC-Aircraft-based model.

It is the technical heart of the repository.

#### Files

- `fgvc-manufacturer-resnet34.ipynb`  
  Main training notebook for the larger model.  
  This notebook contains the serious version of the project:
  - dataset handling,
  - preprocessing,
  - progressive training,
  - test-set evaluation,
  - export logic.

- `fgvc_manufacturer_resnet34_optimized_fastai.pkl`  
  Exported FastAI learner corresponding to the optimized training pipeline.

- `model.pth`  
  Raw PyTorch weights exported for deployment portability.

- `labels.json`  
  Label vocabulary used by the deployed model.

- `test_metrics_summary.csv`  
  Main test-set metrics.

- `test_metrics_summary_tta.csv`  
  Test-set metrics after test-time augmentation.

- `test_per_class_report.csv`  
  Per-class evaluation table.

- `test_best_classes.csv`  
  Best-performing classes.

- `test_worst_classes.csv`  
  Worst-performing classes.

- `test_top_confusions.csv`  
  Most frequent confusion pairs.

- `test_predictions_top5.csv`  
  Per-image prediction table with top-k outputs.

- `confusion_matrix_row_normalized_test.png`  
  Row-normalized confusion matrix for qualitative inspection.

#### Role in the project

This folder contains the material that demonstrates:
- the shift from a small prototype to a serious fine-grained ML pipeline,
- the engineering decisions taken to improve the model,
- the evaluation methodology used to understand failure modes,
- the artifacts required to reproduce or audit the model.

---

### `Hugging Face demo/`

This folder contains the **deployment-ready application files** for the main demo.

It is separate from `training/` because deployment has different needs from training:
- fewer files,
- faster startup,
- simplified runtime,
- explicit artifacts.

#### Files

- `app.py`  
  Main inference application for the deployed model.

- `Dockerfile`  
  Container configuration used by the Hugging Face Space.

- `requirements.txt`  
  Python dependencies for the deployed app.

- `model.pth`  
  The deployment model weights.

- `labels.json`  
  Label vocabulary used by the app.

- `supported_families.json`  
  Human-readable list of supported aircraft families shown in the interface.

- `README.md`  
  Hugging Face Space metadata / documentation file.

- `.gitattributes`  
  Git LFS or repository-specific configuration for large files.

#### Role in the project

This folder contains exactly what is needed to run the public demo:
- inference logic,
- model weights,
- metadata,
- runtime configuration.

It should remain minimal and deployment-oriented.

---

## Root-level files

### `.gitattributes`
Repository-level Git configuration, especially useful when large model artifacts are involved.

### `runtime.txt`
Legacy runtime configuration file. Depending on the deployment mode, it may be informative or obsolete. If deployment is fully Docker-based, this file is no longer central.

---

## Main technical ideas used in the larger FGVC-Aircraft model

The larger model is not just “the same notebook with more classes”.  
It includes several design choices intended to improve robustness and performance.

### 1. Benchmark-based dataset choice

The larger model uses **FGVC-Aircraft**, a benchmark specifically designed for fine-grained visual aircraft classification.

This is important because the dataset provides:
- a realistic classification hierarchy,
- structured annotations,
- official train / validation / test splits,
- bounding boxes,
- a recognized benchmark context.

This makes the project much more credible than a custom toy dataset.

---

### 2. Removal of the copyright banner

The FGVC-Aircraft images include a **20-pixel copyright banner** at the bottom.

The training notebook explicitly removes this banner during preprocessing.

This is a small but important detail:
- it removes useless visual noise,
- it prevents the model from learning spurious correlations,
- it shows attention to dataset-specific preprocessing.

---

### 3. Bounding-box-based cropping

Each image contains a bounding box annotation for the aircraft.

The preprocessing pipeline uses these boxes to crop the aircraft more tightly, with a small margin.

This improves the signal-to-noise ratio:
- less background,
- more focus on the aircraft body,
- better preservation of relevant visual details.

This is especially useful in fine-grained recognition, where subtle shape differences matter.

---

### 4. Cached preprocessing

The notebook preprocesses the aircraft images **once** and stores the resulting crops in a cache folder.

This improves the workflow in two ways:
- faster repeated training,
- cleaner separation between preprocessing and learning.

This is a practical engineering choice, not just a convenience.

---

### 5. Progressive resizing

The larger model is trained with **progressive resizing**.

Instead of training immediately at the highest resolution, the notebook moves through stages such as:
- 256×256,
- 384×384,
- 512×512.

This strategy is useful because:
- lower-resolution training is faster,
- the model can first learn coarse structure,
- higher-resolution stages then refine finer visual cues.

For a fine-grained dataset, this is particularly relevant.

---

### 6. Transfer learning with ResNet34

The main pipeline uses a **ResNet34** backbone.

This choice balances:
- model capacity,
- training stability,
- computational cost,
- deployment simplicity.

The point here was not to use the most fashionable architecture possible, but to build a strong and reproducible baseline with a well-understood convolutional backbone.

---

### 7. Regularization and augmentation

The training pipeline includes:
- image augmentation,
- ImageNet normalization,
- MixUp regularization,
- weight decay.

These techniques improve generalization and make the training pipeline more robust.

---

### 8. Mixed precision

The notebook attempts to use **mixed precision** when available.

This improves:
- memory efficiency,
- training speed,
- practicality on GPU notebook environments such as Kaggle.

---

### 9. Proper split logic

The larger notebook follows the correct ML protocol:
- train split for fitting,
- validation split for monitoring and model development,
- test split for final evaluation.

This matters because it prevents the test set from being used as a tuning tool.

---

### 10. Full test-set evaluation

The project does not stop at a validation accuracy screenshot.

The notebook evaluates the model on the **full official test split** and exports:
- accuracy,
- balanced accuracy / macro recall,
- macro precision,
- macro F1-score,
- top-3 accuracy,
- per-class metrics,
- confusion matrix,
- top confusion pairs.

This is essential for a serious ML project because it makes the model’s behavior interpretable.

---

### 11. TTA evaluation

The notebook also evaluates **test-time augmentation (TTA)**.

This provides a direct comparison between:
- standard test inference,
- augmented test inference.

That comparison is useful to understand how sensitive the model is to small input perturbations.

---

### 12. Deployment-oriented export strategy

The notebook exports both:
- a FastAI `.pkl`,
- and a deployment-friendly pair:
  - `model.pth`
  - `labels.json`

This dual export strategy is important:
- `.pkl` is convenient for notebook-side experimentation,
- `.pth + labels.json` is more robust for deployment.

This is a strong practical design choice for moving from Kaggle to Hugging Face Spaces.

---

## Reproducing the project from scratch

This section explains how someone can rebuild the project end to end.

---

## Part A — Reproducing the initial five-class prototype

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Airplanes_v2.git
cd Airplanes_v2
```

### 2. Open the first prototype

Go to:

```text
demo-5-models/airplanes_5_models.ipynb
```

This notebook contains the first-stage classifier.

### 3. Install the dependencies

Create a virtual environment if needed, then install:

```bash
pip install -r demo-5-models/requirements.txt
```

### 4. Run the notebook

Launch Jupyter or use Kaggle / Colab, then run the notebook cell by cell.

### 5. Export the first model

The notebook exports a small model artifact (`airplanes_5_models.pkl`) that can then be used by the lightweight five-class app.

---

## Part B — Reproducing the larger FGVC-Aircraft pipeline

### 1. Open the main notebook

Go to:

```text
training/fgvc-manufacturer-resnet34.ipynb
```

This is the main notebook for the larger pipeline.

### 2. Prepare the environment

Recommended environment:
- Python 3.10+ or 3.11
- PyTorch
- FastAI
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PIL / Pillow

A Kaggle environment is particularly practical because:
- it provides GPU access,
- it is well suited to notebook experimentation,
- it works well with this project’s workflow.

### 3. Download the FGVC-Aircraft dataset

The notebook contains logic to download and cache the dataset automatically from the Oxford VGG source.

If the dataset has already been cached, the notebook reuses the local copy.

### 4. Run the preprocessing stage

The notebook:
- removes the 20-pixel banner,
- applies bounding-box cropping,
- stores processed images in a cache directory.

### 5. Train the model

Run the notebook end to end to:
- build the data pipeline,
- instantiate the learner,
- perform progressive training,
- evaluate on validation,
- evaluate on the full test split.

### 6. Export the artifacts

The notebook exports:
- `fgvc_manufacturer_resnet34_optimized_fastai.pkl`
- `model.pth`
- `labels.json`
- evaluation tables
- confusion matrix image

### 7. Review the metrics

Inspect the files in `training/` to understand:
- overall performance,
- which classes are easier or harder,
- the main confusion pairs.

---

## Part C — Reproducing the Hugging Face deployment

### 1. Go to the deployment folder

```text
Hugging Face demo/
```

### 2. Inspect the files

The deployment folder contains:
- `app.py`
- `Dockerfile`
- `requirements.txt`
- `model.pth`
- `labels.json`
- `supported_families.json`

### 3. Run locally

From that folder, build and run the app according to the deployment mode.

If using Docker:

```bash
docker build -t aircraft-demo .
docker run -p 7860:7860 aircraft-demo
```

Then open:

```text
http://localhost:7860
```

### 4. Deploy to Hugging Face Spaces

Create a Hugging Face Space, then push the contents of the deployment folder to the Space repository.

A common workflow is:
- maintain the “source of truth” in this GitHub repository,
- keep a local clone of the Hugging Face Space repository,
- copy the files from `Hugging Face demo/` into that clone,
- commit and push to Hugging Face.

---

## Engineering lessons from the project

This repository is valuable not only because it contains a working model, but because it documents several real ML engineering lessons.

### 1. A small working prototype is useful
Starting with a tiny but functional system is often the fastest way to learn the pipeline end to end.

### 2. Deployment reveals hidden problems
A model that works in a notebook is not automatically deployment-ready.

### 3. Preprocessing matters as much as architecture
In a fine-grained task, cropping, resizing, and consistency between training and inference can have a major impact.

### 4. Evaluation must go beyond headline accuracy
Confusion matrices, per-class metrics, and test-set exports are necessary to understand the real behavior of a model.

### 5. Scaling a project should be progressive
The five-class notebook was not wasted effort. It was the first step that made the later FGVC-Aircraft system possible.

---

## Suggested reading order for reviewers

If you are reviewing this repository for academic or technical evaluation, the best order is:

1. **Read this master README**
2. **Open `demo-5-models/`** to understand the initial prototype
3. **Open `training/fgvc-manufacturer-resnet34.ipynb`** to inspect the full training pipeline
4. **Review the evaluation exports in `training/`**
5. **Open `Hugging Face demo/`** to inspect deployment logic

This order makes the progression of the project easier to understand.

---

## Current status of the repository

The repository is structured as:
- an initial five-class prototype,
- a larger FGVC-Aircraft training pipeline,
- a deployment-oriented Hugging Face demo,
- a set of exported evaluation artifacts for auditability and reproducibility.

The project therefore presents both:
- the **learning trajectory**,
- and the **final technical deliverables**.

---

## Future directions

Natural next steps for the project include:
- aligning the deployed inference pipeline even more strictly with training preprocessing,
- experimenting with stronger backbones,
- comparing manufacturer, family, and variant objectives more systematically,
- improving deployment UX,
- extending the app with richer interpretability features.

---

## Author’s note

This project is intentionally presented as an evolving engineering effort rather than as a polished black box.

That is precisely what makes it valuable:
- it shows experimentation,
- redesign,
- debugging,
- evaluation,
- deployment,
- and iteration.

In other words, it shows not only that a model was trained, but that an end-to-end machine learning project was genuinely built and improved over time.
