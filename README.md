
# Microsaccade-benchmark Dataset
---

The **Microsaccade-benchmark Dataset** is a high-resolution, event-based dataset designed for **microsaccade detection, classification, and analysis**, developed by researchers at the University of Galway. Microsaccades are small, involuntary eye movements that occur during visual fixation, playing a critical role in **vision research**, **driver monitoring systems (DMS)**, **neuromorphic vision**, and **eye-tracking applications**. This dataset provides both **raw rendered data** (images, gaze vectors, raw event streams) and **preprocessed training-ready files** to support the development of **event-based neural networks** and other algorithms for modeling fine-grained, high-temporal-resolution eye movement patterns. 

This dataset was introduced in a paper accepted at **BMVC 2025**.

![Microsaccade Event Example](https://huggingface.co/datasets/waseemshariff/Microsaccade-benchmark/resolve/main/misc/ms.gif)

---
## Dataset Structure

The dataset is organized into two main folders:

### 1. `Raw/`

This folder contains raw data, including:
- **Rendered eye images** for each microsaccade class.
- **Gaze vector files** indicating the direction of gaze.
- **Raw event streams** recorded at high temporal resolution.

#### Directory Layout

```
Raw/
├── left_eye/
│   ├── 0.5_left/
│   ├── 0.75_left/
│   ├── 1.0_left/
│   ├── 1.25_left/
│   ├── 1.5_left/
│   ├── 1.75_left/
│   └── 2.0_left/
└── right_eye/
    ├── 0.5_right/
    ├── 0.75_right/
    ├── 1.0_right/
    ├── 1.25_right/
    ├── 1.5_right/
    ├── 1.75_right/
    └── 2.0_right/
```

Each class folder includes:
- **Image files** (`.png` or `.jpg`) of the eye region.
- **Text or `.npy` files** containing gaze vectors.
- **Event stream files** (`.npy` or `.txt`) with event data in `[T, X, Y, P]` format.

### 2. `Training/`

This folder contains **preprocessed `.npy` files** ready for training machine learning models.

#### Structure

```
Training/
├── left/
│   ├── train/
│   ├── val/
│   └── test/
└── right/
    ├── train/
    ├── val/
    └── test/
```

Each file in the `train/`, `val/`, or `test/` directories is a serialized Python dictionary saved in `.npy` format.

---

## `.npy` File Format

Each `.npy` file contains:

```python
{
    "data": event_stream,  # Shape: [N, 4] → [T, X, Y, P]
    "target": target_label
}
```

Where:
- **data**: Event stream data
  - **T**: Timestamp (in seconds)
  - **X**: Pixel x-coordinate
  - **Y**: Pixel y-coordinate
  - **P**: Event polarity (1 = ON, 0 or -1 = OFF)
- **label**: String label indicating microsaccade amplitude and eye (e.g., "1.25_left").

### Example: Loading a Sample

```python
import numpy as np

# Load one sample
sample = np.load("Training/left/train/sample_001.npy", allow_pickle=True)

# Extract event stream and label
event_data = sample.item()["data"]  # Shape: [N, 4] -> [T, X, Y, P]
label = sample.item()["target"]


print("Event data shape:", event_data.shape)
print("Label:", label)
```

---

## Dataset Statistics: Event Streams


| Property             | Value         |
|----------------------|---------------|
| Original resolution  | 800 × 600     |
| ROI resolution       | 440 × 300 (center) |
| Total sequences      | 175,000       |
| Left eye sequences   | 87,500        |
| Right eye sequences  | 87,500        |
| Number of classes    | 7             |
| Class distribution   | Evenly split  |

*Synthetic microsaccade dataset details.*

---
## License: cc-by-nc-4.0
This dataset is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. This license allows for non-commercial use of the dataset, provided proper attribution is given to the authors. Adaptations and modifications are permitted for non-commercial purposes. For full details, please review the CC BY-NC 4.0 license and the license file included in the dataset. As the dataset is gated, you must accept the access conditions on Hugging Face to use it.
---

## Citation

If you use this dataset in your research, please cite the following:

```

@dataset{microsaccade_dataset_2025,
  title={Microsaccade Recognition with Event Cameras: A Novel Dataset},
  author={Waseem Shariff and Timothy Hanley and Maciej Stec and Hossein Javidnia and Peter Corcoran},
  year={2025},
  doi={https://doi.org/10.57967/hf/6965},
publisher={Hugging Face},
  note={Presented at BMVC 2025}
}
@inproceedings{Shariff_2025_BMVC,
author    = {Waseem Shariff and Timothy Hanley and Maciej Stec and Hossein Javidnia and Peter Corcoran},
title     = {Benchmarking Microsaccade Recognition with Event Cameras: A Novel Dataset and Evaluation},
booktitle = {36th British Machine Vision Conference 2025, {BMVC} 2025, Sheffield, UK, November 24-27, 2025},
publisher = {BMVA},
year      = {2025},
url       = {https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_288/paper.pdf}
}
@article{microsaccade_benchmarking_2025,
  title={Benchmarking Microsaccade Recognition with Event Cameras: A Novel Dataset and Evaluation},
  author={Shariff, Waseem and Hanley, Timothy and Stec, Maciej and Javidnia, Hossein and Corcoran, Peter},
  journal={arXiv preprint arXiv:2510.24231},
  year={2025}
}

```

## Contact

For questions, updates, or issues related to the dataset, please open a discussion or pull request in the [Community tab](https://huggingface.co/datasets/waseemshariff/Microsaccade-benchmark/discussions) on the Hugging Face dataset page.
