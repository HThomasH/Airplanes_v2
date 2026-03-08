import os, time, traceback
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gradio as gr
import torch
from fastai.vision.all import load_learner, PILImage

MODEL_PATH = "fgvc_manufacturer_resnet34_optimized_fastai.pkl"

print("[startup] loading learner...")
learn = load_learner(MODEL_PATH)
learn.dls.num_workers = 0
learn.model.eval()
torch.set_num_threads(1)
labels = list(learn.dls.vocab)
print(f"[startup] loaded learner. n_classes={len(labels)}")

@torch.inference_mode()
def predict(img_path):
    t0 = time.time()
    try:
        print("[predict] called")
        print(f"[predict] img_path={img_path}")

        if img_path is None:
            raise gr.Error("No image received.")

        img = PILImage.create(img_path)
        print("[predict] PILImage created")

        pred, pred_idx, probs = learn.predict(img)
        print(f"[predict] learn.predict done in {time.time()-t0:.2f}s | pred={pred}")

        k = min(5, len(labels))
        topv, topi = torch.topk(probs, k)
        return {labels[int(i)]: float(v) for v, i in zip(topv, topi)}

    except Exception as e:
        print("[predict] ERROR\n" + traceback.format_exc())
        raise gr.Error(str(e))

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload an aircraft image", sources=["upload"]),
    outputs=gr.Label(num_top_classes=min(5, len(labels)), label="Prediction"),
    title="FGVC-Aircraft — Manufacturer Classifier",
    description="Upload an aircraft photo. The model predicts the manufacturer (trained on FGVC-Aircraft).",
    allow_flagging="never",
)

# IMPORTANT: pas de queue pour debug (évite les jobs bloqués)
demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, ssr_mode=False)