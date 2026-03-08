# Airplanes_v2 --- Aircraft Image Classification Project

## Project overview

This repository documents a personal computer vision project focused on
**aircraft image classification**.

The project evolved progressively through two main stages:

1.  A **small prototype classifier trained on five aircraft models**
2.  A **larger model trained on the FGVC-Aircraft dataset** with a
    broader set of aircraft categories

The goal of this repository is simply to document the process followed
while building and improving the project: starting from a simple
tutorial-based experiment, then gradually extending it into a more
structured machine learning pipeline and a deployed demo.

The repository therefore contains both:

-   the **initial prototype**, which helped validate the basic workflow
-   the **larger FGVC-Aircraft model**, which explores a more
    challenging fine-grained classification task

------------------------------------------------------------------------

# Project development steps

## 1. Initial prototype: a 5-aircraft classifier

The project started with a small image classifier trained on **five
aircraft models**.

This first version was based on the following tutorial:

https://medium.com/data-science/build-your-first-deep-learning-app-within-an-hour-4e80c120e99f

The purpose of this stage was mainly to understand and test the basic
elements of a deep learning workflow:

-   loading image data
-   creating a training pipeline with FastAI
-   training a convolutional neural network using transfer learning
-   exporting the trained model
-   building a minimal inference interface

This prototype provided a working end‑to‑end pipeline and served as a
starting point for the rest of the project.

------------------------------------------------------------------------

## 2. First deployment with Hugging Face Spaces

After the prototype worked locally, the next step was deploying the
model using **Hugging Face Spaces**.

Demo: https://huggingface.co/spaces/HThomasH/Airplanes_v2

This stage required reorganizing the project slightly in order to
separate:

-   training code
-   model artifacts
-   inference code
-   application dependencies

A small application was built using **Gradio**, allowing users to upload
an aircraft image and receive a prediction from the trained model.

This deployment stage helped explore practical aspects of ML projects
such as:

-   packaging model artifacts
-   defining reproducible dependencies
-   structuring a lightweight inference application

------------------------------------------------------------------------

# Limitations

### Image resolution

The model performs significantly better with **high‑resolution images**.

Low‑resolution images or heavily compressed images may remove visual
details that are important for distinguishing aircraft manufacturers.

When sourcing images from Google Images, it is therefore preferable to
**open the image in full resolution** before uploading it to the demo.

### Viewing angle

Predictions are generally more reliable when the aircraft is
photographed:

-   perfectly from the side
-   perfectly from the front

Performance tends to decrease when aircraft are photographed:

-   from above at an angle
-   from below at an angle
-   with strong perspective distortion
