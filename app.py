# app.py
import streamlit as st
from PIL import Image
import torch
from neural_style_transfer import style_transfer, load_image, tensor_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Art Style Transfer")
content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])

if content_file and style_file:
    content_img = load_image(content_file).to(device)
    style_img = load_image(style_file).to(device)

    if st.button("Transfer Style"):
        output = style_transfer(content_img, style_img)
        output_image = tensor_to_image(output)
        st.image(output_image, caption='Stylized Image', use_column_width=True)
