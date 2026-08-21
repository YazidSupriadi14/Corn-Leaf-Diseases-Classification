# Corn Leaf Disease Classification

Image classification project that detects corn leaf disease from a photo — Common Rust,
Gray Leaf Spot, Northern Leaf Blight, or Healthy — comparing a CNN trained from scratch
against transfer learning with MobileNetV2.

## Problem

Corn foliar diseases can significantly reduce crop yield if not caught early. Manual
diagnosis by visual inspection is slow, inconsistent, and depends on expert knowledge
that isn't always accessible in the field. This project builds an image classifier to
make early detection faster and more accessible.

## Approach

Two models were trained and compared on the same dataset and train/val/test split:

1. **CNN (from scratch)** — 4 convolutional blocks (32→64→128→128 filters) with max
   pooling, followed by dense layers with dropout. Trained end-to-end with random
   weight initialization.
2. **MobileNetV2 (transfer learning)** — ImageNet-pretrained backbone, frozen and
   trained with a custom classification head, then fine-tuned on the top layers at a
   lower learning rate.

Both models were converted to **TensorFlow Lite** (including a quantized variant) and
**TensorFlow.js**, and deployed as an interactive demo with **Gradio**.

## Dataset

[Corn Leaf Diseases dataset](https://www.kaggle.com/datasets/yusufmurtaza01/corn-leaf-diseases)
from Kaggle, 4 classes: `Corn__CommonRust`, `Corn__GrayLeafSpot`, `Corn__Healthy`,
`Corn__NorthernLeafBlight`. Split 70% train / 15% validation / 15% test.

## Results

| Model | Test Accuracy | Test Loss | Parameters |
|---|---|---|---|
| CNN (from scratch) | 94.56% | 0.139 | 3,454,660 |
| MobileNetV2 (transfer learning) | 94.73% | 0.126 | 2,422,468 |

**Per-class performance (CNN from scratch):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Common Rust | 0.98 | 0.93 | 0.96 |
| Gray Leaf Spot | 0.81 | 0.84 | 0.82 |
| Healthy | 0.99 | 1.00 | 1.00 |
| Northern Leaf Blight | 0.92 | 0.97 | 0.94 |

Gray Leaf Spot is the hardest class to classify for both models — its early-stage
symptoms visually overlap with other foliar diseases, which is reflected in the lower
precision/recall for that class.

Model size and inference latency were also benchmarked across SavedModel, TF-Lite
(float32), and TF-Lite (quantized) formats — see the notebook for full benchmark numbers.

## Repository contents

```
notebook.ipynb          # full training, evaluation, and experimentation pipeline
app.py                  # Gradio inference app
requirements.txt        # dependencies for the Gradio app
model/                  # CNN scratch checkpoint (SavedModel, TF-Lite, TFJS)
model_mobilenet/        # MobileNetV2 checkpoint (SavedModel, TF-Lite, quantized, TFJS)
```

## Live demo

Try the model directly: *[add Hugging Face Space link here]*

## Author

Muhammad Yazid Supriadi
