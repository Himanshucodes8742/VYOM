# 🌕 LunarAlign: Multi-modal Image Registration for Chandrayaan-2

[![Problem Statement](https://img.shields.io/badge/Problem%20ID-26166-blue)](#)
[![Organization](https://img.shields.io/badge/Organization-ISRO-orange)](#)
[![Theme](https://img.shields.io/badge/Theme-Space%20Technology-black)](#)

## 📑 Table of Contents
1. [Overview](#overview)
2. [Key Challenges](#key-challenges)
3. [Proposed Architecture](#proposed-architecture)
4. [Feasible Roadmap](#feasible-roadmap)
5. [Tech Stack](#tech-stack)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Installation & Usage](#installation--usage)

---

## 🔭 Overview
This repository provides a comprehensive software solution for **Multi-modal, Sun angle, and scale invariant image correspondence**. The objective is to geometrically align diverse lunar optical images acquired by Chandrayaan-2 (OHRC, TMC-2, IIRS) with reference datasets (LROC NAC, SELENE) into a common coordinate system with **sub-pixel accuracy**.

## ⚠️ Key Challenges Addressed
* **Illumination Variation:** Differing sun azimuth and elevation angles drastically change the appearance of craters and shadows.
* **Viewpoint Variation:** Geometric distortions (shift, scale, rotation, perspective) due to different orbiter trajectories and camera angles.
* **Scale Variation:** Massive resolution differences (e.g., OHRC at ~0.25m vs. IIRS at ~80m vs. TMC-2 at 5m).

---

## 🏗 Proposed Architecture
To achieve robust matching across multi-modal lunar imagery, we propose a hybrid pipeline:
1. **Pre-processing:** Geospatial reprojection using GDAL, radiometric normalization, and image pyramiding for scale matching.
2. **Detector-Free Deep Matching:** Utilizing transformer-based local feature matching (e.g., **LoFTR** or **LightGlue**) which excels in texture-poor and illumination-varied regions compared to traditional SIFT/SURF.
3. **Geometric Verification:** Advanced outlier rejection using **MAGSAC++** to handle extreme noise and compute the homography/affine transformation.
4. **Sub-pixel Refinement:** Patch-based normalized cross-correlation (NCC) or phase correlation to refine match points to sub-pixel accuracy.
5. **Uniformity Enforcement:** Grid-based point selection to ensure a uniform distribution of keypoints across the image.

---

## 🗺 Feasible Roadmap (12-Week Plan)

### Phase 1: Data Acquisition & Pre-processing (Weeks 1 - 2)
* [ ] **Task 1:** Download sample datasets for OHRC, TMC-2, IIRS from ISRO PRADAN and LROC/SELENE reference images.
* [ ] **Task 2:** Develop a data loader to handle PDS4/GeoTIFF formats.
* [ ] **Task 3:** Implement geospatial metadata extraction to approximate initial overlap and handle gross scale differences (Pyramiding).
* [ ] **Task 4:** Apply histogram equalization and shadow-enhancement filters to normalize illumination variations.

### Phase 2: Feature Detection & Matching (Weeks 3 - 5)
* [ ] **Task 1:** Implement baseline traditional algorithms (SIFT, ORB) for benchmarking.
* [ ] **Task 2:** Integrate Deep Learning models (LoFTR / SuperPoint + LightGlue) capable of handling severe illumination (Sun angle) and viewpoint changes.
* [ ] **Task 3:** Develop a scale-invariant matching loop that slides over high-resolution images (OHRC) to match with lower-resolution footprints (TMC/IIRS).

### Phase 3: Outlier Rejection & Homography (Weeks 6 - 7)
* [ ] **Task 1:** Implement RANSAC and MAGSAC++ to filter out false correspondences.
* [ ] **Task 2:** Ensure Uniform Distribution: Divide the image into a grid and select top $N$ high-confidence matches per grid cell.
* [ ] **Task 4:** Calculate the transformation matrix (Affine, Perspective, or Thin Plate Spline for non-rigid terrain distortions).

### Phase 4: Sub-pixel Refinement (Week 8)
* [ ] **Task 1:** Isolate matched keypoint patches.
* [ ] **Task 2:** Apply Phase Correlation / Fourier-Mellin transform on patches to find translational shifts at sub-pixel levels.
* [ ] **Task 3:** Update match coordinates with sub-pixel precision.

### Phase 5: Software UI & Packaging (Weeks 9 - 10)
* [ ] **Task 1:** Develop a generic software wrapper (CLI and GUI using PyQt/Streamlit).
* [ ] **Task 2:** Implement features to ingest source and reference images, select algorithm parameters, and visualize the matching lines.
* [ ] **Task 3:** Export module for registered products (saving the warped image as GeoTIFF) and match point CSVs.

### Phase 6: Testing & Evaluation (Weeks 11 - 12)
* [ ] **Task 1:** Compute Evaluation Metrics (RMSE, Inlier match count, Inlier ratio).
* [ ] **Task 2:** Test across edge cases (extreme shadow differences, maximum scale ratio between OHRC and IIRS).
* [ ] **Task 3:** Final code optimization, documentation, and Dockerization.

---

## 🛠 Tech Stack
* **Language:** Python 3.9+
* **Geospatial Processing:** GDAL, Rasterio, PyProj
* **Computer Vision:** OpenCV, Scikit-Image
* **Deep Learning:** PyTorch (for LoFTR / SuperGlue inference), Kornia
* **GUI/Deployment:** PyQt5 / Streamlit, Docker

---

## 📊 Evaluation Metrics
The software will output a dedicated evaluation report containing:
1. **RMSE (Root Mean Square Error):** Measures the geometric distance between transformed source points and reference points. Goal: Sub-pixel (< 1.0).
2. **Inlier Match Count:** Total number of robust, verified matches.
3. **Inlier Ratio:** `(Inlier Matches / Total Matches) * 100`. Higher indicates better feature extractor reliability.
4. **Spatial Distribution Score:** Variance of match points across grid sectors (to ensure uniform distribution).

---
*Built for SIH / ISRO Challenge ID 26166*
