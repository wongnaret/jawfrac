# JawFrac

Welcome to the repository accompanying the paper *Detecting Mandible Fractures in CBCT Scans using Three-Stage Neural Network*.

An Algorithm of the method can be tried out at [Grand Challenge](https://grand-challenge.org/algorithms/jaw-frac-net/). Simply upload a NIfTI file containing a mandible and infer where fractures were detected.

## Model

![Model](docs/model.png)


## Install with conda

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install -r requirements.txt
```

## Dataset Preparation

Of course. Here is the compiled guide in English.

-----

### Step 1: Install dcm2niix

`dcm2niix` is a tool for converting medical images from DICOM to NIfTI format, which is a popular format for neuroimaging analysis.

**How to Install (for macOS)**

If you are using macOS and have Homebrew installed, you can easily install `dcm2niix` with a single command in your Terminal:

```bash
brew install dcm2niix
brew install pigz          
```

For other operating systems like Windows or Linux, you can download the program from [GitHub Releases](https://github.com/rordenlab/dcm2niix/releases) or install it via Conda:

```bash
conda install -c conda-forge dcm2niix
```

### Step 2: Segmentation with ITK-SNAP

`ITK-SNAP` is a software application used for segmenting structures in 3D medical images. It is particularly useful for brain imaging analysis.

**Main Tool:**

  * **ITK-SNAP:** This is the primary tool we will use for segmentation. It provides several sub-tools to make the process easier, such as:
      * **Paintbrush tool:** For manually painting regions of interest.
      * **Polygon tool:** For drawing the boundaries of structures.
      * **Automatic segmentation:** For automatically segmenting structures based on image properties.

**Segmentation Steps:**

1.  **Open NIfTI file:** Open the NIfTI file converted by `dcm2niix`.
2.  **Select tool:** Choose the appropriate tool for the image and the structure you want to segment.
3.  **Perform Segmentation:** Segment the image using the selected tool.
4.  **Save the result:** Save the segmentation result as a NIfTI file.

**ITK-SNAP Video Tutorials:**

  * **[How to use: ITK Snap - YouTube](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3Dk2h39p4-3qA)**
  * **[ITK-SNAP Tutorial: How to trace stroke lesions in T1w images - YouTube](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3DO9t6aV2_3qY)**

### Step 3: Organize Folder Structure and Rename Files

This is the most crucial part. The `jawfrac` project expects a specific file and directory structure. For each patient, you must create a dedicated subfolder.

Inside each patient's folder, there must be **only** these two files:

  * **`scan.nii.gz`**: The 3D CT scan image file, converted from DICOM.
  * **`label.nii.gz`**: The 3D segmentation mask file where the jawbone is labeled with the value `2`.

**Example Directory Structure:**

Assuming you have data for three patients (patient01, patient02, and patient03), the correct folder structure should be as follows:

```
training_dataset/
├── patient01/
│   ├── scan.nii.gz
│   └── label.nii.gz
│
├── patient02/
│   ├── scan.nii.gz
│   └── label.nii.gz
│
└── patient03/
    ├── scan.nii.gz
    └── label.nii.gz
```

----

## Replicate

When trying to replicate this work, the three stages should be trained separately.

### Mandible segmentation

Start with specifying the correct `work_dir` and `root` in the configuration file `jawfrac/config/mandibles.yaml`. The scan and annotation files are expected to be called `scan.nii.gz` and `label.nii.gz`, respectively. Label 2 in the annotation was used for the mandible, please make sure the mandible is annotated with the label 2!

Now run `train_mandibles.py`. The losses and metrics throughout training will be logged to TensorBoard in the working directory. Furthermore, a copy of the configuration file and checkpoints of the best models will also be stored.

Having trained the first stage, the mandible segmentations can be inferred by running `infer_mandibles.py`. If you want to infer different scans than the ones used during training, please specify the location of these scans in the configuration file. The inference results in a `mandible.nii.gz` file with the mandible segmentation as label 1.

### Fracture segmentation

Specify `work_dir`, `checkpoint_path`, and `root` in `jawfrac/config/fractures_linear.yaml` and train the second stage using the inferred mandible segmentation for improved efficiency. After training, results of the model can be visualized by running `infer_fractures_linear.py`.

### Fracture classification

Again, specify `work_dir`, `checkpoint_path`, and `root` in `jawfrac/config/fractures_linear_displaced.yaml` and train the third stage using the mandible and fracture segmentations. This final model can be used for inference with `infer_fractures_linear_displaced.py`.

----

## Cite

```
@article{jawfracnet,
    author = {van Nistelrooij, Niels and Schitter, Sophie and van Lierop, Pieter and El Ghoul, Khalid and K{\"o}nig, Daniela and Hanisch, Marcel and Tel, Alessandro and Xi, Tong and Thiem, Daniel and Smeets, Ralf and Dubois, Leander and Fl{\"u}gge, Tabea and van Ginneken, Bram and Berg{\'e}, Stefaan and Vinayahalingam, Shankeeth},
    year = {2024},
    month = {06},
    title = {Detecting Mandible Fractures in CBCT Scans using Three-Stage Neural Network},
    journal = {Journal of Dental Research},
    doi = {10.1177/00220345241256618}
}
```
