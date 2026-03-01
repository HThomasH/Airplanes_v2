import gradio as gr
import torch
from fastai.vision.all import load_learner, PILImage

# --- Config ---
MODEL_PATH = "fgvc_manufacturer_resnet34_optimized_fastai.pkl"  # adapte si ton fichier a un autre nom

# --- Load model (CPU-friendly for Spaces) ---
learn = load_learner(MODEL_PATH)
learn.dls.num_workers = 0
learn.model.eval()
torch.set_num_threads(1)

# Manufacturers (classes) from the model vocab
manufacturers = list(learn.dls.vocab)
manufacturers_sorted = sorted(manufacturers)

def predict(img):
    """
    img: PIL.Image (via gr.Image(type="pil"))
    returns: dict[label -> probability] for gr.Label
    """
    if img is None:
        return {}

    img = PILImage.create(img)
    with torch.no_grad():
        pred, pred_idx, probs = learn.predict(img)

    # gr.Label expects a dict of {class: score}
    return {manufacturers[i]: float(probs[i]) for i in range(len(manufacturers))}

supported_md = (
    "### Supported manufacturers\n"
    f"Total classes: **{len(manufacturers_sorted)}**\n\n"
    + "\n".join([f"- {m}" for m in manufacturers_sorted])
)

with gr.Blocks(title="FGVC-Aircraft Manufacturer Classifier") as demo:
    gr.Markdown(
        "# FGVC-Aircraft — Manufacturer Classifier\n"
        "Upload an aircraft photo. The model predicts the **manufacturer** (trained on FGVC-Aircraft)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Upload an aircraft image")
            btn = gr.Button("Predict")
        with gr.Column(scale=1):
            out = gr.Label(num_top_classes=min(5, len(manufacturers)), label="Top predictions")

    gr.Markdown(supported_md)

    btn.click(fn=predict, inputs=inp, outputs=out)

demo.queue().launch(server_name="0.0.0.0", server_port=7860)