"""Run: streamlit run app.py"""
import streamlit as st
from PIL import Image

from src.predict import predict
from src.stats import player_stats

st.set_page_config(page_title="Cricketer Face ID", page_icon="🏏")
st.title("🏏 Indian Cricket Team Face Identification")

up = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
if up:
    img = Image.open(up)
    st.image(img, use_container_width=True)
    try:
        best, top = predict(img)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    if not top:
        st.warning("No face detected.")
    else:
        if best:
            st.success(f"{best.replace('_', ' ')}  ({top[0][1]:.1%})")
            s = player_stats(best)
            if s:
                st.subheader("IPL career stats")
                st.table(s)
        else:
            st.warning("Unknown / low confidence")
        st.caption("Top matches")
        for n, p in top:
            st.progress(p, text=f"{n.replace('_', ' ')} - {p:.1%}")
