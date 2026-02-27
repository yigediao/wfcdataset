# WFC Pose Estimation Dataset

A specialized RGB-D dataset for 6D pose estimation of 2×4″ wood studs in Wood Frame Construction (WFC) scenarios.

## Download

The full dataset is hosted on Dropbox:

[Download Dataset](https://www.dropbox.com/scl/fo/jmfefrziwmegrgcna0bkk/AB52JKwEApu1N1qLLDRmdoc?rlkey=j6tipu1tver6s8bxhu9w8drsv&e=2&st=eh75rfpw&dl=0)

---

## Dataset Contents

| File / Folder | Size | Description |
|---------------|------|-------------|
| `raw_rgbd_datasets.zip` | 10.65 GB | Raw RGB-D data (RGB images, depth maps, PLY point clouds) for all 288 samples |
| `annotations.zip` | 461.77 MB | Ground-truth 6D pose annotations for all samples |
| `our_pose_result.zip` | 897.72 MB | Pose estimation results from our method |
| `foundationpose_pose_result.zip` | 237.97 MB | Pose estimation results from FoundationPose |
| `SAM6D_pose_result.zip` | 1.77 GB | Pose estimation results from SAM-6D |
| `megapose_pose_result.zip` | 2.41 GB | Pose estimation results from MegaPose |
| `plane_fiting_baseline/` | — | Plane-fitting baseline code and results |

---

## Demo Video

[![Demo Video](https://img.youtube.com/vi/Lu1Ta6I1sc4/maxresdefault.jpg)](https://www.youtube.com/watch?v=Lu1Ta6I1sc4)

---

## Overview

Existing pose estimation datasets are ill-suited for evaluating algorithms in WFC scenarios. This dataset was created to address that gap, providing high-resolution RGB-D data of 2×4″ wood studs with fully manual ground-truth 6D pose annotations.

- **288** distinct samples
- **2208 × 1242** RGB-D resolution (≈9× denser point clouds than standard 640×480 datasets)
- Per-sample: high-resolution RGB image, depth image, and PLY-format point cloud
- Fully manual ground-truth pose annotation (no ICP propagation or optical flow)

---

## Hardware

| Component | Specification |
|-----------|--------------|
| Camera | ZED 2i stereo camera |
| Environment | Controlled indoor lighting |
| Object | 2×4″ wood stud |

---

## Dataset Structure

Data was collected across **12 spatial orientations** under **3 exposure levels**, with additional intra-scene variations:

### Camera–Object Orientations (12 total)

| Axis | Values |
|------|--------|
| Latitude | 30°, 60°, 90° |
| Longitude | 0°/180°, ±45°, ±90°, ±135° |

> The camera was physically fixed; the object was rotated to simulate different relative poses.

![Dataset Overview](assets/dataset_overview.jpeg)
*Top: Schematic illustration of relative camera–object pose configurations. Bottom: Representative RGB samples under different exposure conditions and partial occlusion.*

### Lighting Conditions (3 per orientation)

| Condition | EV |
|-----------|----|
| Normal light | 0 EV |
| Low light | −1 EV |
| Strong light | +1 EV |

### Intra-Scene Variations

Each scenario includes samples with:
- Variations in visible top surfaces
- Tilted positions
- Partial occlusion (~10% of stud surface)

---

## Ground Truth Labeling

All 288 samples were annotated using a **fully manual, high-precision approach** — unlike datasets such as YCB-Video, where only keyframes are manually labeled and remaining annotations are propagated via optical flow or automatic refinement.

### Labeling Pipeline

1. **3D Object Model**: A 1:1 high-fidelity OBJ model of the target wood stud was created in Blender from precise physical measurements, avoiding the noise introduced by 3D scanning or NeRF-based reconstruction.
2. **Initial Pose Estimate**: A neural network provided an initial 6D pose estimate as a starting point.
3. **Manual Refinement (GUI)**: The pose was refined interactively via a custom Open3D-based labeling tool, with fine-grained control over translation and rotation.
4. **Dual-Domain Verification**: Alignment was verified in both the 3D RGB-D view and the 2D RGB projection (bounding box overlay). A pose was accepted only when tight alignment was confirmed in both domains.

![Ground Truth Labeling GUI](assets/labeling_gui.jpeg)
*Top: Inputs to the labeling tool — RGB-D point cloud, OBJ model, and RGB image. Bottom: 3D alignment GUI (left) and 2D OBB projection overlay (right) used for dual-domain verification.*

---

## Benchmark

### Evaluation Metrics

#### Translational and Rotational Error

Predicted and ground-truth poses are represented as 4×4 homogeneous transformation matrices. The translational error $E_t$ is the Euclidean distance between translation vectors:

$$E_t = \|\mathbf{t}_{gt} - \mathbf{t}_{pred}\|_2$$

The rotational error $E_r$ is derived from the relative rotation matrix $\mathbf{R}_{diff} = \mathbf{R}_{gt}^T \mathbf{R}_{pred}$:

$$E_r = \theta = \cos^{-1}\left(\frac{\mathrm{trace}(\mathbf{R}_{diff}) - 1}{2}\right)$$

#### ADD-S

The ADD-S metric measures geometric alignment quality and is well-suited for symmetric objects. Given a model point set $\mathcal{M} = \{\mathbf{x}_i\}_{i=1}^{N}$:

$$\text{ADD-S} = \frac{1}{|\mathcal{M}|} \sum_{\mathbf{x}_i \in \mathcal{M}} \min_{\mathbf{x}_j \in \mathcal{M}} \|\mathbf{T}_{gt}\mathbf{x}_i - \mathbf{T}_{pred}\mathbf{x}_j\|$$

### Baseline Methods

| Method | Description |
|--------|-------------|
| **Ours** | Our pipeline with point-cloud refinement, iterative optimization, and regression-based correction |
| **FoundationPose** | BOP leaderboard top method; evaluated at 1280×720 due to GPU memory limits at full resolution |
| **SAM-6D** | BOP leaderboard top method; failed to produce valid estimates for 50/288 samples |
| **MegaPose** | BOP leaderboard top method; evaluated with manually labeled bounding boxes |
| **Plane-fitting Baseline** | Geometry-only baseline: dominant plane fitting on segmented point clouds, no learning-based components |

> All learning-based methods used a ZED 2i-provided segmentation or detection pipeline. Our method and Plane-fitting Baseline use YOLOv8-based detection; FoundationPose uses SAM-based segmentation; SAM-6D uses its built-in segmentation pipeline.

### Accuracy Thresholds

Two thresholds were used to reflect different application requirements:

| Standard | Translation | Rotation | ADD-S |
|----------|------------|----------|-------|
| Grasping (baseline) | ≤ 50 mm | ≤ 5° | < 5% of object diameter |
| Industrial assembly (strict) | ≤ 20 mm | ≤ 2° | < 1% of object diameter (10.026 mm) |

### Overall Performance (Table 2)

> Experiments run on: Intel i9-14900 CPU, NVIDIA RTX 4090 GPU, 64 GB RAM. Invalid outputs (e.g., 50 failed SAM-6D samples) are excluded from error statistics. t-statistics computed from paired differences (baseline − Ours).

**Translation Error (mm)**

| Method | Mean | Median | Std Dev | t-statistic | p < 0.01 |
|--------|------|--------|---------|-------------|----------|
| **Ours** | **2.18** | **1.84** | **2.05** | — | — |
| FoundationPose | 10.63 | 10.16 | 4.82 | −27.12 | Yes |
| SAM-6D | 91.59 | 10.20 | 233.77 | −5.90 | Yes |
| MegaPose | 48.88 | 11.97 | 130.82 | −6.06 | Yes |
| Plane-fitting Baseline | 172.65 | 164.70 | 116.77 | −24.79 | Yes |

**Rotation Error (°)**

| Method | Mean | Median | Std Dev | t-statistic | p < 0.01 |
|--------|------|--------|---------|-------------|----------|
| **Ours** | **1.43** | **1.00** | **1.61** | — | — |
| FoundationPose | 7.25 | 5.30 | 6.13 | −15.27 | Yes |
| SAM-6D | 9.18 | 2.90 | 19.86 | −6.00 | Yes |
| MegaPose | 5.16 | 2.01 | 13.27 | −4.72 | Yes |
| Plane-fitting Baseline | 28.28 | 23.93 | 25.66 | −17.71 | Yes |

**ADD-S Error (mm)**

| Method | Mean | Median | Std Dev | t-statistic | p < 0.01 |
|--------|------|--------|---------|-------------|----------|
| **Ours** | **4.80** | **3.89** | **4.44** | — | — |
| FoundationPose | 25.12 | 16.01 | 21.95 | −15.16 | Yes |
| SAM-6D | 83.00 | 11.24 | 213.10 | −5.65 | Yes |
| MegaPose | 37.15 | 10.09 | 95.83 | −5.76 | Yes |
| Plane-fitting Baseline | 286.48 | 307.28 | 152.73 | −31.37 | Yes |

### Valid Pose Rate (Table 3)

| Method | 20 mm / 2° | ADD-S < 1% Diameter | 50 mm / 5° | ADD-S < 5% Diameter |
|--------|-----------|---------------------|-----------|---------------------|
| **Ours** | **73.61%** | **89.58%** | **95.83%** | **100.00%** |
| FoundationPose | 18.06% | 21.53% | 46.53% | 87.85% |
| SAM-6D | 27.43% | 34.03% | 60.07% | 72.92% |
| MegaPose | 39.58% | 48.96% | 77.08% | 88.19% |
| Plane-fitting Baseline | 2.43% | 2.43% | 6.94% | 16.67% |

> Under the strict industrial-level criteria (20 mm / 2° and ADD-S < 1% diameter), our method achieves up to **4.1×** improvement over FoundationPose, **2.7×** over SAM-6D, and **1.8×** over MegaPose.

---

## Code

| Component | Status |
|-----------|--------|
| Ground truth labeling tool (Open3D GUI) | Coming soon |
| Benchmark evaluation code | Coming soon |

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{xie2026wfc,
  title   = {Advancing Robotic Automation in Wood-Framed Construction Using Vision-Driven Adaptive Control},
  author  = {Xie, Chao and Alwisy, Aladdin},
  journal = {Automation in Construction},
  year    = {2026},
  note    = {Accepted, in press}
}
```

> DOI will be added upon publication.

---

## License

<!-- TODO: specify license -->
