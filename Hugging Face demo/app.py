import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from fastai.vision.all import resnet34
from fastai.vision.learner import create_cnn_model


# -----------------------
# Files
# -----------------------
MODEL_PATH = Path("model.pth")
LABELS_PATH = Path("labels.json")
SUPPORTED_FAMILIES_PATH = Path("supported_families.json")

IMG_SIZE = 256
TOPK = 5


# -----------------------
# Load metadata
# -----------------------
def load_labels(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        labels = data
    elif isinstance(data, dict) and "classes" in data and isinstance(data["classes"], list):
        labels = data["classes"]
    elif isinstance(data, dict):
        try:
            labels = [data[str(i)] for i in range(len(data))]
        except Exception:
            labels = [v for k, v in sorted(data.items(), key=lambda kv: kv[0])]
    else:
        raise ValueError("labels.json: unsupported format")

    if not labels:
        raise ValueError("labels.json: empty label list")

    return labels


def load_supported_families(path: Path):
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        families = data
    elif isinstance(data, dict) and "families" in data and isinstance(data["families"], list):
        families = data["families"]
    else:
        raise ValueError("supported_families.json: unsupported format")

    return sorted(set(str(v).strip() for v in families if str(v).strip()))


LABELS = sorted(load_labels(LABELS_PATH))
SUPPORTED_FAMILIES = load_supported_families(SUPPORTED_FAMILIES_PATH)

N_OUT = len(LABELS)


# -----------------------
# Build model
# -----------------------
device = torch.device("cpu")
torch.set_num_threads(1)

model = create_cnn_model(resnet34, n_out=N_OUT).to(device)
state = torch.load(MODEL_PATH, map_location=device)

if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
    state = state["model"]

missing, unexpected = model.load_state_dict(state, strict=False)

if len(unexpected) > 0:
    print(f"[warn] unexpected keys: {unexpected[:10]} ... ({len(unexpected)})")
if len(missing) > 0:
    print(f"[warn] missing keys: {missing[:10]} ... ({len(missing)})")

model.eval()


# -----------------------
# Preprocess
# -----------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img

    side = max(w, h)
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top

    return ImageOps.expand(
        img,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=(0, 0, 0),
    )


def pil_to_tensor_norm(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    img = pad_to_square(img)
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)

    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.unsqueeze(0)


# -----------------------
# Predict
# -----------------------
@torch.inference_mode()
def predict(pil_img: Image.Image):
    if pil_img is None:
        return {}, "No image provided."

    try:
        x = pil_to_tensor_norm(pil_img).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)

        k = min(TOPK, probs.numel())
        vals, idxs = torch.topk(probs, k=k)

        out = {LABELS[i.item()]: float(v.item()) for v, i in zip(vals, idxs)}
        return out, ""
    except Exception as e:
        return {}, f"Prediction error: {type(e).__name__}: {e}"


# -----------------------
# UI helpers
# -----------------------
def to_bulleted_markdown(items):
    if not items:
        return "Not available in this deployment."
    return "\n".join([f"- {item}" for item in items])


supported_manufacturers_md = to_bulleted_markdown(LABELS)
supported_families_md = to_bulleted_markdown(SUPPORTED_FAMILIES)

intro_md = """
Upload an aircraft photo. The model predicts the manufacturer.
"""

dataset_note_md = """
The application was trained on the FGVC-Aircraft dataset.  
Predictions are more likely to work on aircraft types that belong to the dataset coverage listed below.
"""

CSS = """
#imgbox img { object-fit: contain !important; }
"""


# -----------------------
# UI
# -----------------------
with gr.Blocks(css=CSS, title="FGVC-Aircraft — Manufacturer Classifier") as demo:
    gr.Markdown("# FGVC-Aircraft — Manufacturer Classifier")
    gr.Markdown(intro_md)

    with gr.Row():
        inp = gr.Image(
            type="pil",
            image_mode="RGB",
            label="Upload an aircraft image",
            height=420,
            elem_id="imgbox",
        )
        out = gr.Label(num_top_classes=min(TOPK, N_OUT), label="Top predictions")

    err = gr.Markdown(value="", visible=True)
    btn = gr.Button("Predict")
    btn.click(fn=predict, inputs=inp, outputs=[out, err], queue=False)

    gr.Markdown(dataset_note_md)

    with gr.Accordion("Supported manufacturers", open=False):
        gr.Markdown(supported_manufacturers_md)

    with gr.Accordion("Supported aircraft families", open=False):
        gr.Markdown(supported_families_md)


demo.launch(server_name="0.0.0.0", server_port=7860)
