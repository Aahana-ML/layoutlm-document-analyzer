import streamlit as st
from PIL import Image
import tempfile
import os

from inference import run_inference


st.set_page_config(
    page_title="Document Analyzer",
    page_icon="📑"
)

st.sidebar.title("⚠️ Important Note")

st.sidebar.warning(
    "This model is a research/demo model and may "
    "incorrectly classify some document elements. "
    "Please treat the predicted QUESTION and ANSWER "
    "labels as model predictions, not guaranteed results."
)

st.title("📑 Document Analyzer")
st.write("Upload a document and let our LayoutLM model analyze it.")


uploaded_file = st.file_uploader(
    "Upload a document image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Document")
    st.image(image, width="stretch")


    if st.button("🔍 Analyze Document"):

        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        ) as temp_file:

            image.save(temp_file.name)

            image_path = temp_file.name


        with st.spinner("Analyzing document..."):

            result = run_inference(
                image_path
            )


        st.success("Analysis complete! 🎉")


        # Show visualization
        if os.path.exists("prediction.png"):

            st.subheader(
                "LayoutLM Predictions"
            )

            st.image(
                "prediction.png",
                width="stretch"
            )
    


        # Clean temporary file
        os.remove(image_path)

st.markdown("---")

st.caption("Made with guidance by ChatGPT 🤖")