
# Airplanes_v2 — Aircraft Image Classification Project

## Project overview

This repository documents a personal computer vision project focused on **aircraft image classification**.

The project evolved progressively through two main stages:

1. A **small prototype classifier trained on five aircraft models**
2. A **larger model trained on the FGVC‑Aircraft dataset** with a broader set of aircraft categories

The goal of this repository is simply to document the process followed while building and improving the project: starting from a simple tutorial-based experiment, then gradually extending it into a more structured machine learning pipeline and a deployed demo.

The repository therefore contains both:
- the **initial prototype**, which helped validate the basic workflow,
- the **larger FGVC‑Aircraft model**, which explores a more challenging fine‑grained classification task.

---

# Project development steps

## 1. Initial prototype: a 5‑aircraft classifier

The project started with a small image classifier trained on **five aircraft models**.

This first version was based on the following tutorial:

https://medium.com/data-science/build-your-first-deep-learning-app-within-an-hour-4e80c120e99f

The purpose of this stage was mainly to understand and test the basic elements of a deep learning workflow:

- loading image data
- creating a training pipeline with FastAI
- training a convolutional neural network using transfer learning
- exporting the trained model
- building a minimal inference interface

This prototype provided a working end‑to‑end pipeline and served as a starting point for the rest of the project.

---

## 2. First deployment with Hugging Face Spaces

After the prototype worked locally, the next step was deploying the model using **Hugging Face Spaces**. You can find the demo here: https://huggingface.co/spaces/HThomasH/Airplanes_v2

This stage required reorganizing the project slightly in order to separate:

- training code
- model artifacts
- inference code
- application dependencies

A small application was built using **Gradio**, allowing users to upload an aircraft image and receive a prediction from the trained model.

This deployment stage helped explore practical aspects of ML projects such as:

- packaging model artifacts
- defining reproducible dependencies
- structuring a lightweight inference application

---

## 3. Moving to a larger dataset: FGVC‑Aircraft

After experimenting with the small prototype, the next step was to work with a more challenging dataset.

The project was therefore extended to use the **FGVC‑Aircraft dataset**, a benchmark dataset designed for **fine‑grained aircraft recognition**.

Dataset reference: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

Fine‑Grained Visual Classification of Aircraft  
Maji et al., 2013

The FGVC‑Aircraft dataset contains:

- 10,200 aircraft images
- multiple hierarchical label levels
- official train / validation / test splits
- bounding box annotations

Working with this dataset required adapting the training pipeline to handle:

- more categories
- more subtle visual differences between classes
- more structured preprocessing.

---

# Repository structure

```
Airplanes_v2/

demo-5-models/
    Initial small prototype model

training/
    Training notebooks and evaluation results for the FGVC‑Aircraft model

Hugging Face demo/
    Files used for the deployed inference application for the FGVC‑Aircraft model
```

---

# Folder descriptions

## demo‑5‑models

This folder contains the **initial prototype** classifier trained on five aircraft classes.

Files:

- `airplanes_5_models.ipynb`  
  Notebook used to train the first prototype.

- `airplanes_5_models.pkl`  
  Exported FastAI model.

- `app_5_models.py`  
  Simple inference application used in the first demo.

- `requirements.txt`  
  Dependencies for this small demo.

This prototype is intentionally simple and mainly served as a first working example.

---

## training

This folder contains the notebooks and outputs for the larger model trained on the FGVC‑Aircraft dataset.

Key files:

`fgvc-manufacturer-resnet34.ipynb`  
Main training notebook.

`fgvc_manufacturer_resnet34_optimized_fastai.pkl`  
Exported FastAI learner.

`model.pth`  
PyTorch weights used by the deployed model.

`labels.json`  
Mapping between class indices and labels.

Evaluation outputs:

- confusion matrices
- per‑class reports
- top prediction tables
- summary metrics

These files were generated after evaluating the model on the test split.

---

## Hugging Face demo

This folder contains the files used to run the public demo. You can find the demo here: https://huggingface.co/spaces/HThomasH/FGVC-Aircraft_manufacturers

Files:

`app.py`  
Main inference application.

`model.pth`  
Model weights.

`labels.json`  
Class labels.

`supported_families.json`  
List of aircraft families displayed in the interface.

`Dockerfile`  
Environment definition used by the Hugging Face Space.

`requirements.txt`  
Python dependencies.

---

# Training approach (FGVC model)

The training notebook implements several practical steps:

- dataset download and extraction
- cropping aircraft using bounding boxes
- removing the 20‑pixel copyright banner included in the dataset
- resizing images to a consistent resolution
- training a ResNet‑based classifier with FastAI
- exporting the trained model
- evaluating the model on the official test set

Evaluation outputs include:

- overall accuracy
- per‑class metrics
- confusion matrices
- prediction tables

These results are stored in the `training` folder.

---

# Reproducing the project

## Clone the repository

```
git clone https://github.com/<username>/Airplanes_v2.git
cd Airplanes_v2
```

---

## Running the prototype model

```
cd demo-5-models
pip install -r requirements.txt
```

Open the notebook:

```
airplanes_5_models.ipynb
```

Run the cells to train the small prototype model.

---

## Running the FGVC training notebook

Open:

```
training/fgvc-manufacturer-resnet34.ipynb
```

The notebook contains the full pipeline:

- dataset download
- preprocessing
- training
- evaluation
- model export

A GPU environment such as **Kaggle** is recommended for training.

---

## Running the Hugging Face demo locally

From the `Hugging Face demo` folder:

```
pip install -r requirements.txt
python app.py
```

# Summary

This repository contains:

- a small prototype aircraft classifier
- a larger fine‑grained classification experiment using FGVC‑Aircraft
- a deployed demo application

The project documents the steps followed while building and improving the system, from the initial prototype to the larger model and deployment setup.
