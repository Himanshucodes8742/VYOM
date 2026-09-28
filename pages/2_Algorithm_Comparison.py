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
            eval_suite = [
                {"name": "AKAZE (CLAHE only)", "algo": "akaze", "prep": "clahe"},
                {"name": "AKAZE (Photometric)", "algo": "akaze", "prep": "photometric"},
                {"name": "SIFT (CLAHE only)", "algo": "sift", "prep": "clahe"},
                {"name": "SIFT (Photometric)", "algo": "sift", "prep": "photometric"},
                {"name": "RIFT2 (Phase Congruency)", "algo": "rift2", "prep": "clahe"},
                {"name": "Learned Verifier (trained)", "algo": "learned_verifier", "prep": "clahe"},
                {"name": "Learned Descriptor (trained)", "algo": "learned_descriptor", "prep": "clahe"},
                {"name": "Crater Landmarks (trained CNN)", "algo": "crater_landmarks", "prep": "clahe"},
            ]
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            sun_elev = None
            xml_candidates = list(Path("data").glob("*.xml"))
            for xc in xml_candidates:
                try:
                    from registration_engine.metadata import get_sun_elevation
                    sun_elev = get_sun_elevation(str(xc))
                    break
                except Exception:
                    pass
            
            for i, item in enumerate(eval_suite):
                status_text.text(f"Running {item['name']}...")
                res = run_pipeline(
                    source_path,
                    reference_path,
                    algorithm=item["algo"],
                    preprocessing=item["prep"],
                    source_sun_elevation=sun_elev,
                )
                
                if res["success"]:
                    metrics = res["metrics"]
                    results.append({
                        "Configuration": item["name"],
                        "Algorithm": item["algo"].upper(),
                        "Preprocessing": item["prep"],
                        "RMSE": round(metrics["rmse"], 4),
                        "Inliers": metrics["inlier_count"],
                        "Ratio": round(metrics["inlier_ratio"], 4),
                        "Dist. Score": round(metrics["distribution_score"], 4),
                        "Runtime (s)": round(metrics["runtime"], 4)
                    })
                else:
                    st.error(f"{item['name']} failed: {res['error']}")
                
                progress_bar.progress((i + 1) / len(eval_suite))
                
            status_text.text("Done!")
            
            if results:
                st.subheader("Metrics Comparison")
                st.dataframe(results, use_container_width=True)
                
                st.subheader("RMSE Comparison (Lower is Better)")
                fig, ax = plt.subplots(figsize=(10, 5))
                configs = [r["Configuration"] for r in results]
                rmses = [r["RMSE"] for r in results]
                colors = ['#2ca02c' if 'Photometric' in c else '#1f77b4' for c in configs]
                
                ax.bar(configs, rmses, color=colors)
                ax.set_ylabel("RMSE (pixels)")
                ax.set_title("Root Mean Square Error by Configuration")
                plt.xticks(rotation=25, ha='right')
                max_rmse = max(rmses) if rmses else 1
                for idx, v in enumerate(rmses):
                    ax.text(idx, v + (max_rmse * 0.01), f"{v:.2f}", ha='center', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
    else:
        st.warning("Could not find source and reference images in the selected demo folder.")
