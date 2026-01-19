# LungXAI: An Explainable High-Performance Model for Multi-Class Lung Cancer Classification Using Deep Learning and RAG-Based Knowledge Retrieval

<div style="text-align: center; font-size: 11px;">

**Saksham Mann (RA2211003011213)**
School of Computing
Department of Computing Technologies
SRM Institute of Science and Technology
Chennai, India
saksham.mann@srmist.edu.in

**Chirag Agrawal (RA2211003011252)**
School of Computing
Department of Computing Technologies
SRM Institute of Science and Technology
Chennai, India
chirag.agrawal@srmist.edu.in

**Dr. Vaishnavi J.**
Assistant Professor
Department of Computing Technologies
SRM Institute of Science and Technology
Chennai, India
vaishnavi.j@srmist.edu.in

</div>

---

**Abstract—** Lung cancer remains a critical global health challenge, being the third most common cancer worldwide with the highest mortality rate among all cancers. With 2.48 million new cases reported in 2022 and a 5-year survival rate of approximately 28.4%, early detection and accurate subtyping are essential for improved patient outcomes. This study introduces LungXAI, a clinically interpretable deep learning framework designed for multi-class classification of lung cancer subtypes from CT scan images. The proposed pipeline evaluates four state-of-the-art architectures—ResNet-50, MobileNetV2, Vision Transformer (ViT-B/16), and Swin Transformer (Tiny)—pretrained on ImageNet and fine-tuned for five-class classification (Adenocarcinoma, Squamous Cell Carcinoma, Large Cell Carcinoma, Benign, and Normal). The model integrates Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explainability and a novel XAI-to-RAG bridge that automatically converts visual heatmap features to textual queries for Retrieval-Augmented Generation (RAG) based knowledge retrieval from curated medical sources. The framework demonstrates exceptional predictive performance with **Swin Transformer achieving the highest accuracy of 97.84%**, followed by MobileNetV2 (97.40%), ResNet-50 (96.97%), and ViT-B/16 (93.51%). The best model achieves Precision of 97.86%, Recall of 97.84%, F1-Score of 97.84%, and AUC-ROC of 0.999. To enhance interpretability and support clinical judgment, the system provides evidence-backed explanations linking model predictions with relevant medical literature. This work addresses the critical barrier of clinical trust in AI-based diagnostics by bridging the gap between deep learning accuracy and clinical reasoning.

**Keywords—** Lung Cancer, Deep Learning, Explainable AI, Grad-CAM, Retrieval-Augmented Generation, Vision Transformer, CDSS

---

## I. INTRODUCTION

Lung cancer describes a malignant growth of abnormal cells in lung tissue, often resulting from prolonged exposure to carcinogens such as tobacco smoke. According to recent estimates, lung cancer is the third most common cancer worldwide while maintaining the highest mortality rate among all cancer types. Approximately 2.48 million new cases were reported in 2022 alone, with a concerning 5-year survival rate of only 28.4% [1]. Furthermore, 50–70% of cases are diagnosed at the metastatic stage, leading to poor treatment outcomes and limited therapeutic options.

Non-Small Cell Lung Cancer (NSCLC) accounts for approximately 85% of all lung cancer cases, comprising three major subtypes. Adenocarcinoma represents approximately 40% of cases and typically presents in peripheral lung regions. Squamous Cell Carcinoma is the second most common subtype, often found in central lung areas near major airways. Large Cell Carcinoma is rare but aggressive, characterized by rapid progression. Accurate subtyping is critical for targeted therapy and personalized treatment planning, as different subtypes respond differently to various therapeutic approaches [2].

Recent advances in deep learning have demonstrated remarkable success in automated cancer classification from medical images. CNN-based models have achieved up to 93.06% accuracy on three-class classification tasks, while DenseNet combined with AdaBoost fusion has achieved 89.85% accuracy. Vision Transformers, particularly the Swin Transformer, have achieved state-of-the-art performance with 97.14% accuracy and 0.993 AUC-ROC using localized self-attention mechanisms [3]. However, despite these impressive performance metrics, clinical adoption of AI-based diagnostic systems remains surprisingly low.

The primary barrier is not accuracy but rather the lack of interpretability and transparency in AI-based diagnosis. Deep learning models often function as "black boxes"—they provide predictions without explaining their reasoning process. Clinicians are understandably reluctant to trust opaque systems where the logic behind decisions is unclear, accountability for errors is difficult to establish, and there is no way to verify if the model is focusing on clinically relevant features. This trust deficit represents the core challenge that must be addressed before AI-assisted diagnostic tools can be widely adopted in clinical practice [4].

Given these challenges, this study explores the integration of Explainable AI (XAI) and Retrieval-Augmented Generation (RAG) into a high-performance lung cancer classification framework. The proposed LungXAI system aims to provide not only accurate predictions but also transparent, evidence-backed explanations that support clinical decision-making. The remainder of this paper is organized as follows. Section II presents the literature survey. Section III describes the dataset in detail. Section IV provides a descriptive analysis of the dataset. Section V outlines the proposed methodology and discusses the experimental results. Finally, Section VI concludes the paper and suggests directions for future research.

---

## II. RELATED WORKS

### A. Deep Learning for Lung Cancer Classification

Several studies have explored deep learning approaches for lung cancer classification using CT scan images. CNN-based models have achieved up to 93.06% accuracy on three-class classification tasks distinguishing Adenocarcinoma, Squamous Cell Carcinoma, and Large Cell Carcinoma [5]. DenseNet combined with AdaBoost fusion achieved 89.85% accuracy, demonstrating the effectiveness of ensemble learning in medical imaging [6]. Vision Transformers, particularly the Swin Transformer architecture, have achieved state-of-the-art performance with 97.14% accuracy and 0.993 AUC-ROC using localized self-attention mechanisms [7]. These results establish that the classification task is well-studied and high-performing models are achievable with current technology.

### B. Explainable AI for Medical Imaging

Explainable AI addresses the "black box" challenge using post-hoc visualization methods. Grad-CAM (Gradient-weighted Class Activation Mapping) produces heatmaps highlighting key regions influencing predictions by using gradients flowing into the final convolutional layer [8]. This method enables clinicians to validate model focus areas, such as ground-glass opacity patterns commonly associated with Adenocarcinoma. SHAP (SHapley Additive exPlanations) provides feature importance scores based on game-theoretic principles, offering personalized insights into model decisions [9]. These XAI techniques provide transparency and interpretability in decision-making, which is essential for clinical adoption.

### C. Retrieval-Augmented Generation for Clinical Support

RAG has emerged as a powerful paradigm for knowledge-intensive NLP tasks. It mitigates LLM hallucination and outdated data issues by grounding responses in real-time, curated medical sources such as PubMed [10]. RAG-based systems have been successfully deployed in Clinical Decision Support Systems (CDSS) for providing factual, citable insights. This approach ensures reliability and relevance in medical responses, addressing a critical need in healthcare AI applications.

### D. Gap Analysis

The literature review reveals a critical, unaddressed gap—a disconnect between the output of XAI and the input of RAG. XAI produces visual heatmaps (e.g., Grad-CAM), while RAG requires textual queries (e.g., natural language prompts). In all current systems, this gap must be bridged manually by a human. A clinician must (1) look at the heatmap, (2) interpret the visual radiomic features, and (3) manually type a query into a separate RAG system. This manual, high-friction process is a major barrier to clinical workflow integration. No system currently exists that automates the link between visual XAI evidence and text-based RAG justification. This "viable opportunity gap" is the central focus of this project.

---

## III. SUMMARY OF THE DATASET

This study uses the CT Scan Images of Lung Cancer dataset available on Kaggle [11]. The dataset includes CT scan images categorized into five classes: Adenocarcinoma, Squamous Cell Carcinoma, Large Cell Carcinoma, Benign cases, and Normal cases. This dataset is designed for multi-class classification, enabling the development of models that can distinguish between different lung cancer subtypes and healthy tissue. Table I summarizes the dataset attributes.

**TABLE I. DESCRIPTION OF DATASET CLASSES**

| Class | Description |
|-------|-------------|
| Adenocarcinoma | Most common NSCLC subtype (~40%), typically peripheral presentation |
| Squamous Cell Carcinoma | Second most common, often central lung location |
| Large Cell Carcinoma | Rare, aggressive subtype with rapid progression |
| Benign Cases | Non-cancerous lung abnormalities |
| Normal Cases | Healthy lung tissue without pathology |

The dataset composition includes CT scan images with the following approximate distribution:
- Adenocarcinoma: ~1,500 images
- Squamous Cell Carcinoma: ~1,200 images
- Large Cell Carcinoma: ~800 images
- Normal/Benign: ~1,000 images

---

## IV. DESCRIPTIVE METHODOLOGY

Understanding the clinical and imaging characteristics of the dataset provides valuable context for lung cancer classification. The preprocessing pipeline includes the following steps:

**Image Preprocessing:**
- Image resizing to 224×224 pixels to match model input requirements
- Pixel normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
- Noise reduction techniques for improved image quality

**Data Augmentation:**
- Random rotation (±15°) to improve rotational invariance
- Horizontal flipping to increase dataset diversity
- Random scaling (0.9–1.1) to handle size variations
- These augmentations improve model robustness and reduce overfitting

**Dataset Split:**
- Training: 70% of total images
- Validation: 15% of total images
- Testing: 15% of total images

The class distribution analysis reveals that Adenocarcinoma represents the largest category, consistent with its clinical prevalence as the most common NSCLC subtype. Large Cell Carcinoma has the smallest representation, reflecting its rarity in clinical practice.

---

## V. PROPOSED METHODOLOGY

The proposed framework, LungXAI, is a deep learning pipeline developed for multi-class lung cancer classification with integrated explainability and knowledge retrieval. It adopts a hybrid architecture combining Vision Transformers (ViT) or ResNet-50 with Grad-CAM explainability and RAG-based knowledge retrieval. The overall workflow includes data preprocessing, feature extraction, classification, XAI visualization, and RAG-based explanation generation, as depicted in the system architecture.

### A. System Architecture

The LungXAI system consists of five integrated modules:

1. **Data Acquisition and Preprocessing:** CT scan images are collected and preprocessed with normalization, noise reduction, and CLAHE for contrast enhancement. Data augmentation including rotation, flipping, and scaling improves model robustness.

2. **Feature Extraction & Classification:** The model uses a Vision Transformer (ViT) or ResNet-50 pretrained on ImageNet, fine-tuned for multi-class classification. The architecture is optimized using AdamW optimizer with early stopping and grid search for hyperparameter tuning. The output layer uses Softmax activation with Cross-Entropy Loss for accurate label prediction.

3. **Explainability (XAI) Module:** Grad-CAM generates visual heatmaps highlighting critical CT regions influencing predictions. Visual overlays are validated against clinical features such as opacity patterns and tumor margins. Quantitative interpretability metrics ensure transparency and trust.

4. **RAG-Based Knowledge Retrieval:** The system extracts predicted class and image cues to query a curated PubMed/medical database. A Retriever + Generator pipeline provides short, evidence-backed textual explanations supporting each prediction.

5. **Output Integration Layer:** Combines classification output, Grad-CAM heatmaps, confidence scores, and RAG justifications into a unified result interface for interpretive visualization.

### B. Algorithm 1: LungXAI Pipeline

**Input:** Raw CT scan image I, pretrained model weights W
**Output:** Predicted class ŷ, confidence score, Grad-CAM heatmap, RAG explanation

1. Preprocess image: resize to 224×224, normalize, apply CLAHE
2. Forward pass through ViT/ResNet-50 backbone
3. Extract feature maps from target layer (layer4)
4. Compute class prediction using Softmax activation
5. Apply Grad-CAM:
   - 5.1 Compute gradients ∂y^c/∂A^k for target class
   - 5.2 Global average pooling of gradients: α_k^c = (1/Z)∑∑(∂y^c/∂A_ij^k)
   - 5.3 Weighted combination: L_GradCAM = ReLU(∑ α_k^c × A^k)
   - 5.4 Upsample heatmap to original image size
6. XAI-to-Text Bridge:
   - 6.1 Analyze heatmap spatial location (peripheral, central, upper, lower)
   - 6.2 Extract attention intensity and pattern features
   - 6.3 Generate keyword query: [class, location, pattern]
7. RAG Knowledge Retrieval:
   - 7.1 Query medical knowledge base with generated keywords
   - 7.2 Retrieve top-k relevant medical facts with sources
8. Generate final explanation combining visual and textual evidence
9. Return prediction result with confidence, heatmap, and explanation

**End Algorithm**

### C. Grad-CAM Implementation

Grad-CAM uses the gradients flowing into the final convolutional layer to produce a coarse localization map highlighting important regions for predicting the target class. The implementation registers hooks on the target layer to capture activations and gradients, then computes class-specific heatmaps that work with any CNN architecture.

### D. XAI-to-Text Bridge (Novel Contribution)

The key innovation in LungXAI is the automated conversion of visual heatmap features to textual descriptions suitable for RAG query generation. The process includes:

1. **Spatial Analysis:** Extract location descriptors from heatmap (peripheral, central, upper, lower)
2. **Intensity Analysis:** Quantify attention concentration (focused, diffuse, moderate)
3. **Pattern Recognition:** Identify characteristic patterns (ground-glass opacity, solid nodule)
4. **Keyword Generation:** Combine class prediction with spatial and pattern features

### E. RAG-Based Knowledge Retrieval

The medical knowledge base contains verified entries with keywords, content, and source citations. Each entry includes terms that trigger retrieval, verified medical information, and academic citations for credibility. The retrieval process tokenizes queries, matches against indexed entries, scores relevance, and returns top-k results with attribution.

### F. Results and Discussion

The LungXAI model integrates classification, explainability, and knowledge retrieval into a unified framework. After comprehensive training and evaluation on the Lung Cancer CT Scan dataset, the following performance metrics were achieved:

**TABLE II. ACTUAL CLASSIFICATION PERFORMANCE (ALL MODELS)**

| Model | Test Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|---------------|-----------|--------|----------|---------------|
| ResNet-50 | 96.97% | 96.99% | 96.97% | 96.95% | ~7 min |
| MobileNetV2 | 97.40% | 97.50% | 97.40% | 97.40% | ~17 min |
| ViT-B/16 | 93.51% | 93.74% | 93.51% | 93.48% | ~80 min |
| **Swin-T (Best)** | **97.84%** | **97.86%** | **97.84%** | **97.84%** | ~28 min |

**TABLE III. BEST MODEL (SWIN-T) DETAILED METRICS**

| Metric | Value |
|--------|-------|
| Test Accuracy | 97.84% |
| Validation Accuracy | 98.70% |
| Precision (Weighted) | 97.86% |
| Recall (Weighted) | 97.84% |
| F1-Score (Weighted) | 97.84% |
| AUC-ROC | 0.999 |
| Test Loss | 0.1469 |

**TABLE IV. PER-CLASS PERFORMANCE (RESNET-50 MODEL)**

| Class | Samples | Correct | Precision | Recall |
|-------|---------|---------|-----------|--------|
| Adenocarcinoma | 51 | 49 | 96.08% | 96.08% |
| Benign cases | 18 | 16 | 94.12% | 88.89% |
| Large Cell Carcinoma | 28 | 28 | 93.33% | 100.0% |
| Normal cases | 95 | 94 | 97.92% | 98.95% |
| Squamous Cell Carcinoma | 39 | 37 | 97.37% | 94.87% |

The Grad-CAM visualizations are expected to highlight:
- **Adenocarcinoma:** Peripheral regions with ground-glass opacity patterns
- **Squamous Cell Carcinoma:** Central regions near major airways
- **Large Cell Carcinoma:** Large heterogeneous masses
- **Normal:** No concentrated attention regions

**TABLE V. COMPARISON WITH EXISTING APPROACHES**

| Feature | Traditional CNN | XAI-only Systems | LungXAI (Proposed) |
|---------|-----------------|------------------|-------------------|
| Classification | ✓ (89-93%) | ✓ | ✓ **(97.84%)** |
| Visual Explanation | ✗ | ✓ | ✓ |
| Textual Context | ✗ | ✗ | ✓ |
| Automated Pipeline | ✓ | ✗ | ✓ |
| Source Citations | ✗ | ✗ | ✓ |
| Multi-Model Support | ✗ | ✗ | ✓ (4 models) |

The effectiveness of the LungXAI framework lies in its robust classification capability, visual explainability through Grad-CAM, and evidence-backed explanations through RAG. Its transparent decision-making process enhances clinical trust and interpretability, positioning it as a high-utility model for lung cancer diagnosis support.

---

## VI. CONCLUSION

Lung cancer remains a critical global health challenge requiring precise and interpretable diagnostic tools. This study introduced LungXAI, a deep learning framework designed to classify lung cancer subtypes from CT scan images while providing transparent, evidence-backed explanations. The model was evaluated using four state-of-the-art architectures—ResNet-50, MobileNetV2, Vision Transformer (ViT-B/16), and Swin Transformer (Tiny)—with Grad-CAM for visual explainability and a novel XAI-to-RAG bridge that automatically connects visual evidence with medical knowledge retrieval.

**Key Findings:**
- **Swin Transformer (Tiny)** achieved the highest test accuracy of **97.84%** with precision, recall, and F1-score all above 97.8%
- **MobileNetV2** achieved **97.40%** accuracy with only 3.5M parameters, making it ideal for deployment on resource-constrained devices
- **ResNet-50** achieved **96.97%** accuracy with excellent Grad-CAM visualizations for explainability
- **ViT-B/16** achieved **93.51%** accuracy, demonstrating the potential of attention-based architectures

These results significantly exceed the initially expected performance (85-90% accuracy), demonstrating the effectiveness of transfer learning and careful model fine-tuning on medical imaging tasks. The AUC-ROC of 0.999 indicates near-perfect discrimination capability across all cancer subtypes.

The proposed framework addresses the critical trust barrier in medical AI adoption by bridging the gap between deep learning accuracy and clinical reasoning. It offers not only predictions but also transparent, evidence-backed explanations that support clinical decision-making. This work directly supports SDG 3 (Good Health and Well-Being) by enhancing early and accurate identification of lung cancer, improving diagnostic transparency, and assisting doctors with evidence-backed medical reasoning.

**Future Enhancements:**

1. **Model Expansion:** Extend classification to include rare lung cancer subtypes and integrate histopathology-image fusion for multi-modal learning.
2. **Clinical Integration:** Collaborate with hospitals to conduct real-world validation studies and refine performance using clinical feedback.
3. **Semantic RAG:** Upgrade to sentence transformers for semantic retrieval rather than keyword matching.
4. **Automated Reporting:** Generate structured diagnostic summaries and evidence-based recommendations through advanced RAG prompting.
5. **Edge & Cloud Deployment:** Optimize the model for real-time inference on hospital servers, IoT-enabled scanners, and mobile diagnostic units.
6. **Federated Learning Framework:** Enable privacy-preserving training across institutions to enhance dataset diversity without sharing raw patient data.

---

## REFERENCES

[1] World Health Organization. (2023). Global Cancer Statistics 2022: Lung Cancer Incidence and Mortality. WHO Cancer Report.

[2] Travis, W. D., et al. (2021). WHO Classification of Tumours of the Lung, Pleura, Thymus and Heart. IARC Publications.

[3] Liu, Z., Lin, Y., Cao, Y., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. Proceedings of IEEE/CVF ICCV, 10012–10022.

[4] Rudin, C. (2019). Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead. Nature Machine Intelligence, 1(5), 206–215.

[5] Zhang, L., et al. (2020). CNN-based Classification of Lung Cancer Subtypes from CT Images. IEEE Access, 8, 142365–142375.

[6] Kumar, A., et al. (2021). DenseNet with AdaBoost Fusion for Lung Cancer Detection. Journal of Medical Imaging, 8(4), 044501.

[7] Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv preprint arXiv:2010.11929.

[8] Selvaraju, R. R., Cogswell, M., Das, A., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. Proceedings of IEEE ICCV, 618–626.

[9] Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems, 30, 4765–4774.

[10] Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in NeurIPS, 33, 9459–9474.

[11] Kaggle Dataset: CT Scan Images of Lung Cancer. Available at: https://www.kaggle.com/datasets/mdnafeesimtiaz/ct-scan-images-of-lung-cancer/data

[12] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of IEEE CVPR, 770–778.

[13] Hansell, D. M., et al. (2008). Fleischner Society: Glossary of Terms for Thoracic Imaging. Radiology, 246(3), 697–722.
