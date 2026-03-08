import gradio as gr
import torch
from fastai.vision.all import load_learner, PILImage

MODEL_PATH = "airplane_recognition_model.pkl"

learn = load_learner(MODEL_PATH)
learn.dls.num_workers = 0     
learn.model.eval()
torch.set_num_threads(1)

labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)
    with torch.no_grad():
        pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an airplane image"),
    outputs=gr.Label(num_top_classes=min(5, len(labels)), label="Prediction"),
    title="Airplane Recognition",
    description="FastAI airplane classifier deployed on Hugging Face Spaces. Please upload an image from one of the following planes: Dassault Rafale, Piper Cub, Boeing 737, McDonnell Douglas C-17, Citation XL."
)

demo.queue().launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
