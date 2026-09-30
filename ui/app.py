import os, requests, streamlit as st
from PIL import Image

API = os.getenv("API_URL", "http://localhost:8000")

st.title("🖼 DeepVision — классификация изображений")
st.caption("ResNet-18 обученный на CIFAR-10")

file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])
if file:
    img = Image.open(file)
    st.image(img, caption="Input", use_column_width=True)
    with st.spinner("Инференс..."):
        r = requests.post(f"{API}/predict", files={"file": file.getvalue()})
    if r.ok:
        for p in r.json()["predictions"]:
            st.progress(p["confidence"], text=f"{p['class']}: {p['confidence']:.3f}")
    else:
        st.error(r.text)
