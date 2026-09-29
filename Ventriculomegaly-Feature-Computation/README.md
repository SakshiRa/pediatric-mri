<div align="justify">

# A Ventriculomegaly Feature Computational Framework for CT

This repository provides the computational framework to evaluate a set of eight ventriculomegaly features on non-contrast Computed Tomography (CT) scans. **The main contribution of this repository is the feature computation pipeline**, which extracts the following clinical metrics: the Evans indices (x, y, z), Callosal Angle (CA), normalized maximum third ventricle width (NMax-3VW), cerebrospinal fluid to brain volume ratio (CSF2BVR), the maximum eccentricity of lateral ventricles (MaxEccLV), and a proxy Splenial Angle (p-SA). 

## Prerequisites: AC-PC & Midline Alignment
**Important:** This framework assumes that your input CT volumes are already preprocessed and tilt-corrected such that the Anterior Commissure to Posterior Commissure (AC-PC) line is strictly horizontal and the midline is strictly vertical. 

This pipeline works best when utilized alongside our previously open-sourced tool: **[AC-PC Prediction Framework for Head CT](https://github.com/samadanilab/AC-PC-Landmark-Prediction)**. It is highly recommended that you process your scans through that pipeline first to automatically predict the AC and PC coordinates, as these landmarks are strictly required by this repository to guide ROI placement and slice selection.

To facilitate the head tilt correction, we will include a Jupyter Notebook to perform the AC-PC and midline alignment using 3D-Slicer, shortly to this repository. 
* **Note on Midline Alignment:** Because our previous framework does not include midline localization, users must manually provide a set of 20 mid-sagittal-plane points to achieve midline alignment. This manual localization step takes under one minute per scan and is required before running the provided alignment notebook.

## Dataset Context & Compatibility
This framework's feature extraction pipelines were evaluated on a structurally diverse set of 200+ non-contrast CT scans from patients with Normal Pressure Hydrocephalus (NPH), Alzheimer's Disease (AD), post-traumatic volume loss (PTVL), and healthy controls (HC). 

Hyperparameters for the image processing pipelines were empirically derived and proven to work robustly on this structurally diverse set. Because our scans primarily included chronic neurodegeneration, these parameters are expected to translate well to datasets without severe acute findings (such as significant midline shifts, large bleeds, or mass effects). 

## Data Setup
**This repository does not contain head CT data**. Before running the pipeline, make a copy of `config.template.json`, rename it to `config.json`, and update the paths to point to your local dataset.

## System Requirements
Our evaluations were performed on a Linux machine (Ubuntu 24.04.3 LTS) with the following specifications:
* **CPU:** Intel(R) Core(TM) i9-10850K CPU @ 3.60GHz 20M Cache
* **Cores:** 10 Cores / 20 Threads
* **Memory:** 32GB

## Illustration of Computed Features
Below are representative CT scans of patients with NPH, AD, HC, and PTVL showcasing the computed ventriculomegaly features.

![Ventriculomegaly-Feature-Computation](assets/Figure2_KeyMeasures.JPG)

## Data Availability Disclaimer
As this framework was originally developed using data from the Dept. of Veterans Affairs (VA), we are unable to publicly release the internal VA head CT scans due to patient privacy regulations and strict data security policies. However, we provide the complete source code and a detailed step-by-step tutorial to enable researchers to extract these features on their own datasets.

## Citation
If you find this code useful, please cite our work:
> Kadaba Sridhar S, Kuang R, Dysterheft Robb J, Samadani U. A ventriculomegaly feature computational pipeline to improve the screening of normal pressure hydrocephalus on CT. J Neurosurg. 2024 Mar 8;141(3):822-832. doi: 10.3171/2023.12.JNS231780. PMID: 38457801.

## Acknowledgments
This work utilized **3D Slicer** (https://www.slicer.org/) for initial image preprocessing and head tilt correction analysis. If you use this code for your research, please also cite 3D Slicer in your work.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

</div>
