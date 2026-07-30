import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json

st.set_page_config(page_title="Geo-Dashboard", layout="wide")

CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('checkpoints/finetuned_resnet18_day4.pt', map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_thresholds():
    with open('checkpoints/thresholds.json') as f:
        return json.load(f)

model = load_model()
thresholds = load_thresholds()

embedding_model = models.resnet18(weights=None)
embedding_model.fc = nn.Linear(embedding_model.fc.in_features, 10)
embedding_model.load_state_dict(model.state_dict())
embedding_model.fc = nn.Identity()
embedding_model.eval()

st.title("Satellite Geo-Dashboard")
st.write("Upload a before and after tile to classify land-use and detect change.")

threshold_mode = st.radio(
    "Sensitivity mode",
    options=["high_recall", "balanced", "high_precision"],
    index=1,
    format_func=lambda x: {"high_recall": "High Recall (catch more changes)",
                            "balanced": "Balanced",
                            "high_precision": "High Precision (fewer false alarms)"}[x],
    horizontal=True
)
active_threshold = thresholds[threshold_mode]
st.caption(f"Active threshold: {active_threshold:.3f}")

col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Before (T1)", type=['jpg', 'jpeg', 'png'])
with col2:
    file2 = st.file_uploader("After (T2)", type=['jpg', 'jpeg', 'png'])

if file1 and file2:
    img1 = Image.open(file1).convert('RGB')
    img2 = Image.open(file2).convert('RGB')

    t1 = transform(img1).unsqueeze(0)
    t2 = transform(img2).unsqueeze(0)

    with torch.no_grad():
        logits1 = model(t1)
        logits2 = model(t2)
        probs1 = torch.softmax(logits1, dim=1)[0]
        probs2 = torch.softmax(logits2, dim=1)[0]
        pred1 = probs1.argmax().item()
        pred2 = probs2.argmax().item()

        emb1 = embedding_model(t1)
        emb2 = embedding_model(t2)
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()

    changed = similarity < active_threshold

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img1, caption="T1 (Before)", use_container_width=True)
        st.metric("Predicted class", CLASSES[pred1])
        st.metric("Confidence", f"{probs1[pred1].item():.1%}")
    with c2:
        st.image(img2, caption="T2 (After)", use_container_width=True)
        st.metric("Predicted class", CLASSES[pred2])
        st.metric("Confidence", f"{probs2[pred2].item():.1%}")
    with c3:
        diff = np.abs(np.array(img1.resize((64,64))).astype(int) - np.array(img2.resize((64,64))).astype(int)).astype('uint8')
        st.image(diff, caption="Pixel Difference Heatmap", use_container_width=True)
        st.metric("Cosine Similarity", f"{similarity:.3f}")
        if changed:
            st.error("CHANGE DETECTED")
        else:
            st.success("No significant change")
else:
    st.info("Upload both images to see results.")