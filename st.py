import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A

st.set_page_config(page_title="Horizon Detection", layout="wide")

st.title("🌅 Horizon Line Detection using UNet")

# Upload model file from user
uploaded_model = st.file_uploader("Upload your model file (.pth)", type=["pth"])

if uploaded_model is None:
    st.info("Please upload your model file to continue.")
    st.stop()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.dropout = nn.Dropout2d(0.5)
        
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        # Bottleneck
        bn = self.bottleneck(self.pool4(d4))
        bn = self.dropout(bn)  # Apply dropout at bottleneck for regularization

        # Decoder with skip connections
        up1 = self.up1(bn)
        up1 = torch.cat([up1, d4], dim=1)
        up1 = self.conv1(up1)

        up2 = self.up2(up1)
        up2 = torch.cat([up2, d3], dim=1)
        up2 = self.conv2(up2)

        up3 = self.up3(up2)
        up3 = torch.cat([up3, d2], dim=1)
        up3 = self.conv3(up3)

        up4 = self.up4(up3)
        up4 = torch.cat([up4, d1], dim=1)
        up4 = self.conv4(up4)

        return self.final_conv(up4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_file):
    try:
        # Save uploaded model file temporarily
        with open("temp_model.pth", "wb") as f:
            f.write(model_file.getbuffer())
        model = UNet(in_channels=3, out_channels=1)
        # Load model state dict
        model.load_state_dict(torch.load("temp_model.pth", map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(uploaded_model)
if model is None:
    st.stop()
else:
    st.success("Model loaded successfully! 🎉")

# Albumentations preprocessing pipeline
transform = A.Compose([
    A.Resize(288, 384),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

def preprocess_image(image: Image.Image):
    image_np = np.array(image.convert("RGB"))
    augmented = transform(image=image_np)
    tensor = augmented['image'].unsqueeze(0).to(device).float()
    return tensor, image_np

def predict_mask(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob_mask = torch.sigmoid(output)
        binary_mask = (prob_mask > 0.5).float()
    return binary_mask.squeeze().cpu().numpy()

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
        cv2.line(line_image, horizon_points[i-1], horizon_points[i], line_color, 2)
    return line_image

uploaded_images = st.file_uploader("Upload image(s) to infer", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_images:
    for img_file in uploaded_images:
        try:
            img = Image.open(img_file)
            img = img.resize((384, 288))
            input_tensor, img_np = preprocess_image(img)
            mask = predict_mask(input_tensor)
            binary_mask = (mask * 255).astype(np.uint8)
            mask_overlay = overlay_mask_on_image(img_np, mask)
            horizon_overlay = draw_horizon_line(img_np, mask)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.subheader("Uploaded Image")
                st.image(img)
            with c2:
                st.subheader("Predicted Mask")
                st.image(binary_mask, clamp=True, channels="GRAY")
            with c3:
                st.subheader("Mask Overlay")
                st.image(mask_overlay)
            with c4:
                st.subheader("Horizon Line Overlay")
                st.image(horizon_overlay)
        except Exception as e:
            st.error(f"Error processing {img_file.name}: {e}")
else:
    st.info("Please upload one or more images for inference.")

with st.expander("Instructions"):
    st.markdown("""
    1. Upload a trained UNet model file (.pth).
    2. Upload image(s) in PNG, JPG, or JPEG format.
    3. The app will display the original image, predicted mask, mask overlay, and horizon line overlay.
    """)
