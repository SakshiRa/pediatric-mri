# MRI-Adapted Ventriculomegaly Feature Extraction

Quantitative biomarkers for ventriculomegaly from T1-weighted brain MRI, adapted from the CT-based pipeline in [Samadani et al., Journal of Neurosurgery (2024)](https://thejns.org/view/journals/j-neurosurg/141/3/article-p822.xml).

## Features

| Notebook | Measure | View | Description |
|----------|---------|------|-------------|
| `CSF2BVR_MRI` | CSF-to-Brain Volume Ratio | Axial (full volume) | Ratio of total CSF to brain volume |
| `EvansX_MRI` | Evans Index X | Axial | Max frontal horn width / inner skull diameter |
| `EvansY_MRI` | Evans Index Y | Axial | Frontal horn anterior-posterior length ratio |
| `EvansZ_MRI` | Evans Index Z | Coronal | Frontal horn height ratio |
| `NMax3VW_MRI` | Normalized Max 3rd Ventricle Width | Axial | Max 3rd ventricle width between AC and PC |
| `CallosalAngle_MRI` | Callosal Angle | Coronal | Angle between lateral ventricle medial walls at PC |
| `MaxEccLV_MRI` | Max Eccentricity of Lateral Ventricles | Coronal | Ellipse eccentricity of lateral ventricle cross-sections |
| `ProxySplenialAngle_MRI` | Proxy Splenial Angle | Sagittal + Axial | Angle of occipital horn orientation |

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/SakshiRa/pediatric-mri.git
cd pediatric-mri
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager), then:

```bash
uv sync
```

This creates a `.venv/` and installs all dependencies from `pyproject.toml`.

### 2. Add your data

Place MRI scans under `data/` (gitignored). Two datasets are currently supported:

**OASIS-1** (atlas-registered):
```
data/oasis1/disc1/
  OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/*_masked_gfc.img
  OAS1_0002_MR1/...
```

**ABCD** (single-volume T1w):
```
data/abcd/
  sub-XXXXX_ses-XX_T1w_acpc.nii.gz
  sub-XXXXX_ses-XX_acpc_coords.txt    # AC/PC coordinates from ART
```

### 3. Run a notebook

```bash
uv run jupyter notebook EvansX_MRI.ipynb
```

Each notebook has the same structure:
1. **Configuration** -- set `DATASET = "oasis1"` or `"abcd"`
2. **Single-subject test** -- runs on one scan with visualization
3. **Batch processing** -- runs on all scans, saves CSV to `output/`

## CT-to-MRI Adaptation

The original CT pipeline used Hounsfield Unit (HU) windowing to separate tissue types. T1-weighted MRI has arbitrary intensity ranges (e.g., 0-3126 for OASIS), so the key adaptations are:

- **No HU windowing** -- removed `window_stack_sitk()` entirely
- **`normalize_to_uint8()`** -- linearly maps each slice's intensity range to 0-255 for OpenCV thresholding functions
- **Otsu + adaptive thresholding** -- works as-is since it operates on histogram shape, not absolute values
- **Floodfill from (0,0)** -- unchanged for brain mask extraction

## Project Structure

```
pediatric-mri/
  mri_utils.py              # Shared utilities (orientation, segmentation, contour tracing)
  CSF2BVR_MRI.ipynb         # CSF-to-brain volume ratio
  EvansX_MRI.ipynb           # Evans Index X
  EvansY_MRI.ipynb           # Evans Index Y
  EvansZ_MRI.ipynb           # Evans Index Z
  NMax3VW_MRI.ipynb          # Normalized max 3rd ventricle width
  CallosalAngle_MRI.ipynb    # Callosal angle
  MaxEccLV_MRI.ipynb         # Max eccentricity of lateral ventricles
  ProxySplenialAngle_MRI.ipynb  # Proxy splenial angle
  pyproject.toml             # Python dependencies
  data/                      # MRI scans (gitignored)
  output/                    # Results CSVs (gitignored)
```

## AC/PC Coordinates

All features reference the **Anterior Commissure (AC)** and/or **Posterior Commissure (PC)** as anatomical landmarks:

- **OASIS-1**: Hardcoded in T88 atlas voxel space -- `AC = (85, 110, 73)`, `PC = (85, 87, 73)`
- **ABCD**: Per-subject coordinates loaded from `_acpc_coords.txt` files (output of Charlie's ART script)

## Dependencies

- Python 3.11+
- NumPy, Pandas, OpenCV, SimpleITK, scikit-image, matplotlib, statsmodels

All managed via `uv sync` -- see `pyproject.toml`.

## Citation

If you use this pipeline, please cite both this repository and the original paper:

```bibtex
@software{rathi2024mri_ventriculomegaly,
  author    = {Rathi, Sakshi and Samadani, Uzma},
  title     = {MRI-Adapted Ventriculomegaly Feature Extraction Pipeline},
  url       = {https://github.com/SakshiRa/pediatric-mri},
  year      = {2024}
}

@article{kadabasridhar2024ventriculomegaly,
  author    = {Kadaba Sridhar, Sharada and Kuang, Rui and Dysterheft Robb, Jen and Samadani, Uzma},
  title     = {A ventriculomegaly feature computational pipeline to improve the screening of normal pressure hydrocephalus on {CT}},
  journal   = {Journal of Neurosurgery},
  year      = {2024},
  volume    = {141},
  number    = {3},
  pages     = {822--832},
  doi       = {10.3171/2023.12.JNS231780}
}
```

Kadaba Sridhar S, Kuang R, Dysterheft Robb J, Samadani U. *A ventriculomegaly feature computational pipeline to improve the screening of normal pressure hydrocephalus on CT.* Journal of Neurosurgery. 2024;141(3):822-832. doi:[10.3171/2023.12.JNS231780](https://doi.org/10.3171/2023.12.JNS231780)
