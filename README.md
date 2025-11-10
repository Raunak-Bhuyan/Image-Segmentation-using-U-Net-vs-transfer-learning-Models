# Comparative Analysis: U-Net vs Transfer Learning Models for Biomedical Image Segmentation  
*(Project conducted during SN Bose Summer Research Internship)*

## 📌 Overview  
This project addresses the problem of **automating segmentation of cell nuclei** in biomedical microscope images using the [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018) dataset.  
Segmentation at pixel-level is crucial in medical imaging because it preserves spatial context, which is often lost in standard classification models. We built and evaluated:  
- A custom **U-Net** architecture implemented from scratch with Keras  
- Two transfer learning-based pipelines (using **VGG16** and **ResNet-50**) for comparison  
The goal is to determine the most effective approach for high-precision segmentation in a biomedical context.

---

## 🎯 Objectives  
- Compare the segmentation performance of U-Net and transfer learning models (VGG16, ResNet-50) using metrics like **Accuracy**, **IoU (Intersection over Union)** and **Dice Coefficient**  
- Investigate training efficiency, convergence speed, and generalization under limited data  
- Assess robustness and interpretability of models in medical imaging settings  

---

## 🛠 Technologies & Tools Used  
- **Programming Language**: Python  
- **Deep Learning Framework**: TensorFlow & Keras  
- **Data Handling & Augmentation**: NumPy, OpenCV, scikit-image  
- **Classical ML**: scikit-learn (Random Forest classifier in the VGG16 pipeline)  
- **Monitoring & Utilities**: TensorBoard, ModelCheckpoint, EarlyStopping  
- **Hardware**: GPU-enabled environment (e.g., Google Colab or HPC)  
- **Dataset**: Data Science Bowl 2018 — cell nuclei segmentation  

---

## 🚀 Implementation  
### Preprocessing  
- All images were resized to a common resolution and normalized for consistent input  
- Data augmentation included rotations, horizontal/vertical flips, zooms, and shifts to improve generalization  
- Masks were converted to binary format for segmentation targets  

### Model Architectures  
- **U-Net**: Built from scratch in Keras featuring encoder–decoder structure with skip connections to preserve spatial detail  
- **VGG16 + Random Forest**: Used pre-trained VGG16 as a feature extractor (weights frozen) and trained a Random Forest classifier on pixel-wise features for segmentation  
- **ResNet-50**: Evaluated as a transfer learning baseline for segmentation performance  

### Training Setup  
- Loss function: Binary Cross Entropy + Dice Loss (for U-Net)  
- Optimizer: Adam  
- Callbacks: ModelCheckpoint to save best models, EarlyStopping to avoid overfitting, TensorBoard for real-time monitoring  
- Metrics tracked: Accuracy, IoU, Dice Coefficient  
- Training/Validation split: Standard hold-out from the dataset with stratified sampling to preserve class balance  

---

## 📊 Results Summary  
| Model                         | Accuracy | IoU Score | Epochs to Best | Notes                                        |
|------------------------------|----------|-----------|----------------|----------------------------------------------|
| U-Net (custom)               | ~85%     | ~0.78     | ~50            | Best performance; fine spatial detail captured |
| VGG16 + Random Forest        | ~79%     | ~0.71     | ~30            | Good baseline; missed finer boundaries       |
| ResNet-50                    | ~81%     | ~0.73     | ~35            | Strong general model; less tailored for segmentation |

**Key observations**:  
- U-Net’s skip connections played a significant role in retaining boundary and shape information of nuclei.  
- Transfer learning models (VGG16, ResNet-50) provided decent segmentation but lacked precision in capturing fine details.  
- Hybrid deep-learning + classical ML pipeline (VGG16 + Random Forest) demonstrates a creative alternative but requires further tuning for pixel-level tasks.


---

## 📥 How to Run  
1. Clone this repository:  
   ```bash
   git clone https://github.com/Raunak-Bhuyan/Image-Segmentation-using-U-Net-vs-transfer-learning-Models.git
   cd Image-Segmentation-using-U-Net-vs-transfer-learning-Models
   
2. Install required libraries (example):
   pip install tensorflow keras numpy opencv-python scikit-learn matplotlib seaborn

3. Download and extract the Data Science Bowl 2018 dataset, then organize it under /datasets/ with your preprocessed images & masks.

4. Run a specific model script:
   python U-net_train_test.py
   or
    python VGG.py
   or
    python Resnet50.py

## Conclusion

This project clearly indicates that purpose-built segmentation architectures like U-Net outperform off-the-shelf transfer learning models when it comes to pixel-level biomedical image segmentation. For tasks that demand fine spatial detail—such as cell nucleus segmentation—the encoder-decoder architecture with skip connections is especially beneficial.

## Acknowledgements & Contact

Project undertaken during the SN Bose Summer Research Internship.
For any questions or feedback, feel free to reach out at: gauravbanik2005@gmail.com


---
Thank you for exploring this project!
 
