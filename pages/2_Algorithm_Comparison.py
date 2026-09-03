import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

from registration_engine.pipeline import run_pipeline
from registration_engine.io_utils import list_demo_pairs

st.set_page_config(page_title="Algorithm Comparison", layout="wide")
st.title("Algorithm Comparison")
st.write("Compare the performance of SIFT, AKAZE, and RIFT2 on a selected demo pair.")

DATA_DIR = Path("data/demo_pairs")
demo_pairs = list_demo_pairs(DATA_DIR)

if not demo_pairs:
    st.warning("No demo pairs found in data/demo_pairs.")
else:
    selected_input = st.selectbox("Choose demo pair:", demo_pairs)
    
    pair_dir = DATA_DIR / selected_input
    src_candidates = list(pair_dir.glob("source.*")) + list(pair_dir.glob("ohrc.*"))
    ref_candidates = list(pair_dir.glob("target.*")) + list(pair_dir.glob("reference.*")) + list(pair_dir.glob("nac.*"))
    
    if src_candidates and ref_candidates:
        source_path = str(src_candidates[0])
        reference_path = str(ref_candidates[0])
        st.info(f"Using demo images: {src_candidates[0].name} and {ref_candidates[0].name}")
        
        if st.button("Run all algorithms", type="primary"):
            algorithms = ["sift", "akaze", "rift2"]
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, algo in enumerate(algorithms):
                status_text.text(f"Running {algo.upper()}...")
                res = run_pipeline(source_path, reference_path, algorithm=algo)
                
                if res["success"]:
                    metrics = res["metrics"]
                    results.append({
                        "Algorithm": algo.upper(),
                        "RMSE": round(metrics["rmse"], 4),
                        "Inliers": metrics["inlier_count"],
                        "Ratio": round(metrics["inlier_ratio"], 4),
                        "Dist. Score": round(metrics["distribution_score"], 4),
                        "Runtime (s)": round(metrics["runtime"], 4)
                    })
                else:
                    st.error(f"{algo.upper()} failed: {res['error']}")
                
                progress_bar.progress((i + 1) / len(algorithms))
                
            status_text.text("Done!")
            
            if results:
                st.subheader("Metrics Comparison")
                st.dataframe(results, use_container_width=True)
                
                st.subheader("RMSE Comparison")
                fig, ax = plt.subplots(figsize=(8, 5))
                algos = [r["Algorithm"] for r in results]
                rmses = [r["RMSE"] for r in results]
                
                ax.bar(algos, rmses, color=['#1f77b4', '#2ca02c', '#ff7f0e'])
                ax.set_ylabel("RMSE (pixels)")
                ax.set_title("Root Mean Square Error by Algorithm")
                max_rmse = max(rmses) if rmses else 1
                for i, v in enumerate(rmses):
                    ax.text(i, v + (max_rmse * 0.01), f"{v:.2f}", ha='center', fontweight='bold')
                st.pyplot(fig)
                
    else:
        st.warning("Could not find source and reference images in the selected demo folder.")
