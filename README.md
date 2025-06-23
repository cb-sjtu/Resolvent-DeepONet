# Resolvent-DeepONet

Resolvent-DeepONet: A Hybrid Framework for Predicting Energy Fluctuations in Wall-Bounded Turbulence Across Reynolds Numbers

## Overview

Resolvent-DeepONet is a hybrid framework combining analytical methods and deep learning techniques to predict energy fluctuations in wall-bounded turbulence across different Reynolds numbers. The framework consists of two main components:

1. **Resolvent Mode Generation**: Generate optimal modes describing the main characteristics of turbulence based on analytical methods.
2. **Machine Learning Prediction**: Use the generated mode data to predict energy fluctuations through deep learning models such as FNN, Resolvent-CNN, and Resolvent-DeepONet.

All code is developed based on [DEEPXDE](https://deepxde.readthedocs.io/en/latest/user/cite_deepxde.html).

---

## Code Structure

### 1. Resolvent Mode Generation (Matlab)
- `RunResolvent.m`: Generate optimal modes for turbulence data.
- `diff2.m`, `diff2_2nd.m`, `diff6.m`: Auxiliary functions for derivative calculations.
- `chebdif.m`, `clencurt.m`, `clencurtStretch.m`: Functions for computing Chebyshev nodes and weights.
- `Load_DNSProfile.m`: Load DNS data and interpolate it onto Chebyshev grids.

### 2. Machine Learning Prediction (Python)
#### Data Processing
- `src/getdata.py`: Load and preprocess data, including normalization and reshaping.
- Data files are stored in the `data_gen` folder.

#### Model Definition
- `src/network.py`: Define the deep learning network structure, including branch and trunk networks.
- `module.py`: Define multiple-input operator networks (MIONet) and other variants.

#### Model Training and Prediction
- `src/Rsolvent-DeepONet.py`: Main training script for the Resolvent-DeepONet model.
  - Build the network using `DeepONet_resolvent_jiami`.
  - Train the model and save results.
  - Post-processing includes prediction and saving results.
- `src/postprocess.py`: Save prediction results and call plotting functions.
- `src/plot_results.py`: Plot images of prediction results.

---

## Steps to Run

### 1. Resolvent Mode Generation
Run the Matlab file `RunResolvent.m` to generate optimal mode data. This step processes turbulence data and saves the generated mode files.

```matlab
RunResolvent
```

The generated mode files will be saved in the `data_gen` folder.

---

### 2. Machine Learning Prediction
Run the following Python scripts for training and prediction:

#### Resolvent-DeepONet Model
Run the `src/Rsolvent-DeepONet.py` file for training and prediction:

```bash
python src/Rsolvent-DeepONet.py
```

After training, the model will be saved to the specified path, and loss curves and prediction results will be generated.

---

## Notes

- Before running the code for different models, ensure that the `net` module file from DEEPXDE is replaced with the project's `module.py`.
- Data files should be stored in the `data_gen` folder, and the paths should be correct.
- Matlab and Python scripts need to be run separately to complete the entire process.

---

## Citation

If you use this project, please cite the relevant literature for DEEPXDE:
[DEEPXDE](https://deepxde.readthedocs.io/en/latest/user/cite_deepxde.html)
