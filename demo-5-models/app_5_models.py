# ------------------------------------------------------------
# Import required libraries
# ------------------------------------------------------------
# Gradio is used to build the web interface for the demo.
import gradio as gr

# PyTorch is used as the deep learning backend for the model.
import torch

# FastAI utilities for loading the trained model and handling images.
from fastai.vision.all import load_learner, PILImage


# ------------------------------------------------------------
# Load the trained model
# ------------------------------------------------------------
# Path to the serialized FastAI model produced during training.
MODEL_PATH = "airplane_recognition_model.pkl"

# load_learner restores the full FastAI Learner object, which includes:
# - the trained neural network
# - the preprocessing pipeline
# - the label vocabulary
learn = load_learner(MODEL_PATH)

# Disable multiprocessing workers for the dataloader.
# This avoids potential issues in constrained environments such as
# Hugging Face Spaces where parallel workers may cause crashes.
learn.dls.num_workers = 0

# Put the model in evaluation mode.
# This disables training-specific behaviors such as dropout
# and ensures deterministic inference.
learn.model.eval()

# Limit PyTorch to a single CPU thread.
# This improves stability and avoids resource contention
# in lightweight deployment environments.
torch.set_num_threads(1)


# ------------------------------------------------------------
# Retrieve class labels
# ------------------------------------------------------------
# The vocabulary stored in the FastAI dataloaders contains the
# list of classes the model was trained to recognize.
labels = learn.dls.vocab


# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------
# This function is called by the Gradio interface every time
# a user uploads an image and clicks "Predict".
def predict(img):

    # Convert the uploaded image into a FastAI-compatible PILImage object.
    img = PILImage.create(img)

    # Disable gradient computation during inference to improve performance.
    with torch.no_grad():

        # learn.predict performs the full inference pipeline:
        # - preprocessing of the input image
        # - forward pass through the neural network
        # - computation of class probabilities
        pred, pred_idx, probs = learn.predict(img)

    # Return the probabilities for each class in dictionary form.
    # Gradio's Label component expects a mapping {label: probability}.
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


# ------------------------------------------------------------
# Build the Gradio interface
# ------------------------------------------------------------
# gr.Interface provides a simple way to expose the model
# through a web-based interactive interface.
demo = gr.Interface(

    # Function executed when the user submits an image
    fn=predict,

    # Image upload component
    inputs=gr.Image(
        type="pil",
        label="Upload an airplane image"
    ),

    # Output component showing the predicted classes
    outputs=gr.Label(
        num_top_classes=min(5, len(labels)),
        label="Prediction"
    ),

    # Title displayed at the top of the interface
    title="Airplane Recognition",

    # Short description explaining the purpose of the demo
    description=(
        "FastAI airplane classifier deployed on Hugging Face Spaces. "
        "Please upload an image from one of the following planes: "
        "Dassault Rafale, Piper Cub, Boeing 737, "
        "McDonnell Douglas C-17, Citation XL."
    )
)


# ------------------------------------------------------------
# Launch the web application
# ------------------------------------------------------------
# queue() enables request queuing so multiple users can use the
# interface simultaneously without crashing the application.
#
# launch() starts the Gradio server. The parameters ensure that
# the app runs correctly inside the Hugging Face Spaces container.
demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssr_mode=False
)
