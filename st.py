import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A
import gdown 
import os
import sys

st.set_page_config(page_title="Horizon Detection", layout="wide")

MODEL_PATH = "final_unet_horizon.pth"
GDRIVE_FILE_ID = "1N0LGqvlwpgmOdky9aXwFUsOUSytBT8Oq"



# Load your model (ensure model weights path is correct)
from model import UNet  # You must define your UNet class in model.py or inline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title(sys.version)

@st.cache_resource
def load_model():
    os.makedirs("t1", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights from Google Drive...")
        url = "https://drive.google.com/file/d/1N0LGqvlwpgmOdky9aXwFUsOUSytBT8Oq/view?usp=drive_link"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Preprocessing: same as used in validation
transform = A.Compose([
    A.Resize(288, 384),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

def preprocess_image(image: Image.Image):
    image = np.array(image.convert("RGB"))
    augmented = transform(image=image)
    tensor_image = augmented['image'].unsqueeze(0).to(device).float()
    return tensor_image, image

def predict_mask(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()
    return pred.squeeze().cpu().numpy()

def overlay_mask_on_image(image_np, mask_np, color=(0, 255, 0), alpha=0.5):
    overlay = image_np.copy()
    mask_colored = np.zeros_like(image_np)
    mask_colored[mask_np > 0.5] = color
    cv2.addWeighted(mask_colored, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

def draw_horizon_line(image_np, mask_np, line_color=(255, 0, 0)):
    h, w = mask_np.shape
    horizon_points = []

    for col in range(w):
        column = mask_np[:, col]
        indices = np.where(column > 0.5)[0]
        if len(indices) > 0:
            top_y = indices[0]
            horizon_points.append((col, top_y))

    line_image = image_np.copy()
    for i in range(1, len(horizon_points)):
        cv2.line(line_image, horizon_points[i - 1], horizon_points[i], line_color, 2)

    return line_image

# Streamlit UI
# st.set_page_config(page_title="Horizon Detection", layout="wide")
st.title("🌅 Horizon Line Detection using UNet")

uploaded_files = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        image_tensor, image_np = preprocess_image(image)
        mask = predict_mask(image_tensor)

        # Mask binary image
        binary_mask = (mask * 255).astype(np.uint8)

        # Overlays
        mask_overlay = overlay_mask_on_image(image_np, mask)
        horizon_overlay = draw_horizon_line(image_np, mask)

        col1, col2, col3, col4 = st.columns(4)


        with col1:
            st.subheader("Uploaded Image")
            st.image(image)

        with col2:
            st.subheader("Predicted Mask")
            st.image(binary_mask, clamp=True, channels="GRAY")

        with col3:
            st.subheader("Mask Overlay")
            st.image(mask_overlay)

        with col4:
            st.subheader("Horizon Line Overlay")
            st.image(horizon_overlay)
