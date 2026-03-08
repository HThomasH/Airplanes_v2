import json
from pathlib import Path

import gradio as gr
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

IMG_SIZE = 256  # d'après ton training (Resize Pad -> 256)
TOPK = 5


# -----------------------
# Load labels
# -----------------------
def load_labels(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    # accepte ["Airbus", ...] ou {"0":"Airbus", ...} ou {"classes":[...]}
    if isinstance(data, list):
        labels = data
    elif isinstance(data, dict) and "classes" in data and isinstance(data["classes"], list):
        labels = data["classes"]
    elif isinstance(data, dict):
        # dict index->label (string keys)
        try:
            labels = [data[str(i)] for i in range(len(data))]
        except Exception:
            # fallback : valeurs triées par clé
            labels = [v for k, v in sorted(data.items(), key=lambda kv: kv[0])]
    else:
        raise ValueError("labels.json: format non supporté")
    if not labels:
        raise ValueError("labels.json: liste vide")
    return labels


LABELS = load_labels(LABELS_PATH)
N_OUT = len(LABELS)


# -----------------------
# Build model (FastAI arch, no pickle)
# -----------------------
device = torch.device("cpu")
torch.set_num_threads(1)

model = create_cnn_model(resnet34, n_out=N_OUT).to(device)
state = torch.load(MODEL_PATH, map_location=device)

# ton fichier peut être soit un state_dict direct, soit un dict {"model": state_dict}
if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
    state = state["model"]

missing, unexpected = model.load_state_dict(state, strict=False)
# strict=False pour être robuste, mais on garde un check minimal
if len(unexpected) > 0:
    print(f"[warn] unexpected keys: {unexpected[:10]} ... ({len(unexpected)})")
if len(missing) > 0:
    print(f"[warn] missing keys: {missing[:10]} ... ({len(missing)})")

model.eval()


# -----------------------
# Preprocess: pad-to-square -> resize -> tensor -> normalize(ImageNet)
# -----------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pad_to_square(img: Image.Image) -> Image.Image:
    # Pad symétrique avec des pixels noirs (comme pad_mode='zeros')
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))


def pil_to_tensor_norm(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    img = pad_to_square(img)
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)

    x = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.unsqueeze(0)  # (1,3,H,W)


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
# UI (no flicker)
# -----------------------
supported_md = "\n".join([f"- {m}" for m in LABELS])

CSS = """
#imgbox img { object-fit: contain !important; }
"""

with gr.Blocks(css=CSS, title="FGVC-Aircraft — Manufacturer Classifier") as demo:
    gr.Markdown("# FGVC-Aircraft — Manufacturer Classifier")
    gr.Markdown("Upload an aircraft photo. The model predicts the manufacturer.")
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

    gr.Markdown("## Supported manufacturers")
    gr.Markdown(supported_md)

# Important sur Spaces: server_name 0.0.0.0 + port 7860
demo.launch(server_name="0.0.0.0", server_port=7860)