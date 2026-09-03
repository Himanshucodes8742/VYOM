import streamlit as st
import os
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
import io
import csv

from registration_engine.pipeline import run_pipeline
from registration_engine.io_utils import list_demo_pairs

st.set_page_config(page_title="Lunar Image Registration", layout="wide")

st.title("Lunar Image Registration")
st.write("A tool for registering and aligning lunar surface images. Select a demo pair or upload your own images to align the source image onto the reference frame using feature matching and homography.")

st.subheader("Input Selection")

DATA_DIR = Path("data/demo_pairs")
demo_pairs = list_demo_pairs(DATA_DIR)

demo_pairs.insert(0, "Upload my own images")

selected_input = st.selectbox("Choose input data:", demo_pairs)

source_path = None
reference_path = None

if selected_input == "Upload my own images":
    col1, col2 = st.columns(2)
    with col1:
        source_upload = st.file_uploader("Upload Source Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    with col2:
        reference_upload = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
else:
    pair_dir = DATA_DIR / selected_input
    src_candidates = list(pair_dir.glob("source.*")) + list(pair_dir.glob("ohrc.*"))
    ref_candidates = list(pair_dir.glob("target.*")) + list(pair_dir.glob("reference.*")) + list(pair_dir.glob("nac.*"))
    
    if src_candidates and ref_candidates:
        source_path = str(src_candidates[0])
        reference_path = str(ref_candidates[0])
        st.info(f"Using demo images: {src_candidates[0].name} and {ref_candidates[0].name}")
    else:
        st.warning("Could not find source and reference images in the selected demo folder.")

algorithm = st.selectbox("Feature Detection Algorithm:", ["sift", "akaze", "rift2"])

if st.button("Register images", type="primary"):
    ready = False
    temp_dir = None
    
    if selected_input == "Upload my own images":
        if source_upload is not None and reference_upload is not None:
            temp_dir = tempfile.TemporaryDirectory()
            source_path = os.path.join(temp_dir.name, source_upload.name)
            reference_path = os.path.join(temp_dir.name, reference_upload.name)
            with open(source_path, "wb") as f:
                f.write(source_upload.getvalue())
            with open(reference_path, "wb") as f:
                f.write(reference_upload.getvalue())
            ready = True
        else:
            st.error("Please upload both source and reference images.")
    else:
        if source_path and reference_path:
            ready = True

    if ready:
        with st.spinner("Resampling... Preprocessing... Detecting and matching features... Filtering outliers... Warping image... Computing metrics..."):
            result = run_pipeline(source_path, reference_path, algorithm=algorithm)
            
        if not result["success"]:
            st.error(result["error"])
        else:
            st.success("Registration successful!")
            
            # Metrics
            metrics = result["metrics"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("RMSE (pixels)", f"{metrics['rmse']:.4f}")
            m2.metric("Inlier Count", metrics["inlier_count"])
            m3.metric("Inlier Ratio", f"{metrics['inlier_ratio']:.4f}")
            m4.metric("Distribution Score", f"{metrics['distribution_score']:.4f}")
            
            # Images
            st.subheader("Image Comparison")
            c1, c2, c3 = st.columns(3)
            src_img = Image.open(source_path)
            ref_img = Image.open(reference_path)
            warped_img = result["warped_image"]
            
            c1.image(src_img, caption="Source Image", use_container_width=True)
            c2.image(ref_img, caption="Reference Image", use_container_width=True)
            c3.image(warped_img, caption="Registered (Warped) Source", use_container_width=True)
            
            # Scatter Plot
            st.subheader("Match Points on Reference Image")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(ref_img, cmap="gray")
            
            matches = result["matches"]
            kp_source = result["kp_source"]
            kp_reference = result["kp_reference"]
            
            src_pts = []
            ref_pts = []
            for m in matches:
                src_pts.append(kp_source[m.queryIdx].pt)
                ref_pts.append(kp_reference[m.trainIdx].pt)
                
            if ref_pts:
                rx, ry = zip(*ref_pts)
                ax.scatter(rx, ry, c="red", s=5, alpha=0.7, label="Inlier Matches")
                ax.legend()
                
            ax.axis("off")
            st.pyplot(fig)
            
            # Downloads
            st.subheader("Downloads")
            d1, d2, d3 = st.columns(3)
            
            img_buf = io.BytesIO()
            Image.fromarray(warped_img).save(img_buf, format="PNG")
            d1.download_button("Download Warped Image (PNG)", data=img_buf.getvalue(), file_name="warped_image.png", mime="image/png")
            
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(["source_x", "source_y", "reference_x", "reference_y"])
            for (sx, sy), (rx, ry) in zip(src_pts, ref_pts):
                writer.writerow([sx, sy, rx, ry])
            d2.download_button("Download Match Points (CSV)", data=csv_buf.getvalue(), file_name="match_points.csv", mime="text/csv")
            
            json_str = json.dumps(metrics, indent=4)
            d3.download_button("Download Metrics (JSON)", data=json_str, file_name="metrics.json", mime="application/json")
            
        if temp_dir is not None:
            temp_dir.cleanup()
