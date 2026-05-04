import importlib
import os
import pathlib
import sys
import tempfile
from pathlib import Path


def configure_runtime() -> None:
    mpl_dir = Path(tempfile.gettempdir()) / "airplanes_v2_mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath


configure_runtime()

import gradio as gr
import torch
from fastai.vision.all import PILImage, load_learner

MODEL_PATH = Path(__file__).with_name("airplane_recognition_model.pkl")
LEGACY_PLUM_MODULES = (
    "alias",
    "autoreload",
    "bear",
    "dispatcher",
    "function",
    "method",
    "overload",
    "parametric",
    "promotion",
    "repr",
    "resolver",
    "signature",
    "type",
    "util",
    "version",
)


def install_pickle_compat() -> None:
    # The exported FastAI learner references older `plum.*` module paths.
    for suffix in LEGACY_PLUM_MODULES:
        old_name = f"plum.{suffix}"
        new_name = f"plum._{suffix}"
        try:
            sys.modules.setdefault(old_name, importlib.import_module(new_name))
        except ModuleNotFoundError:
            continue


install_pickle_compat()

learn = load_learner(MODEL_PATH)
learn.dls.num_workers = 0
learn.model.eval()
torch.set_num_threads(1)
labels = list(learn.dls.vocab)


def predict(img):
    img = PILImage.create(img)
    with torch.no_grad():
        _, _, probs = learn.predict(img)
    return {label: float(prob) for label, prob in zip(labels, probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an airplane image"),
    outputs=gr.Label(num_top_classes=min(5, len(labels)), label="Prediction"),
    title="Airplane Recognition",
    description="FastAI airplane classifier deployed on Hugging Face Spaces. Please upload an image from one of the following planes: Dassault Rafale, Piper Cub, Boeing 737, McDonnell Douglas C-17, Citation XL.",
)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
