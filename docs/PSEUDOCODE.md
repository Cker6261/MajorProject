# Explainable AI for Lung Cancer Classification
## Project Pseudocode & Algorithm

---

## Main Pipeline Algorithm

```
ALGORITHM: ExplainableLungCancerClassification

INPUT:  CT scan image (chest CT slice)
OUTPUT: Prediction with visual and textual explanation

FUNCTION lung_cancer_pipeline(input_image):
    
    # ===============================================================
    # PHASE 1: IMAGE PREPROCESSING
    # ===============================================================
    
    image <- load_image(input_image)
    image <- resize(image, size=224x224)
    image <- normalize(image, mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
    tensor <- convert_to_tensor(image)
    
    
    # ===============================================================
    # PHASE 2: MODEL PREDICTION (MobileNetV2 - CNN)
    # ===============================================================
    
    logits <- MobileNetV2_Model(tensor)
    probabilities <- softmax(logits)
    prediction <- argmax(probabilities)
    confidence <- max(probabilities)
    
    class_name <- CLASS_NAMES[prediction]
    # CLASS_NAMES = [adenocarcinoma, benign, large_cell, normal, squamous]
    
    
    # ===============================================================
    # PHASE 3: GRAD-CAM VISUALIZATION (XAI)
    # ===============================================================
    
    # Get activations from last convolutional layer (features[18])
    activations <- forward_hook(model.features[18], tensor)
    
    # Compute gradients via backpropagation
    target_score <- logits[prediction]
    gradients <- backward(target_score)
    
    # Compute importance weights (Global Average Pooling)
    weights <- global_average_pool(gradients)
    
    # Generate weighted activation map
    heatmap <- ReLU(SUM(weights * activations))
    heatmap <- normalize(heatmap, range=[0, 1])
    heatmap <- resize(heatmap, size=224x224)
    
    
    # ===============================================================
    # PHASE 4: HEATMAP ANALYSIS
    # ===============================================================
    
    # Analyze spatial distribution
    attention_regions <- find_high_attention_regions(heatmap, threshold=0.5)
    spatial_location <- classify_location(attention_regions)
    # locations: upper_left, upper_right, lower_left, lower_right, central
    
    # Analyze intensity
    intensity_score <- mean(heatmap[attention_regions])
    coverage <- count(heatmap > 0.5) / total_pixels
    focus_score <- sum(top_10_percent_pixels) / sum(all_pixels)
    
    # Generate visual description
    visual_cues <- {
        "location": spatial_location,
        "intensity": intensity_score,
        "coverage": coverage,
        "focus_score": focus_score,
        "pattern": detect_pattern(attention_regions)
    }
    
    
    # ===============================================================
    # PHASE 5: SEMANTIC RAG-BASED KNOWLEDGE RETRIEVAL
    # ===============================================================
    
    # Generate semantic query
    query_text <- f"{class_name} {spatial_location} {pattern}"
    query_embedding <- sentence_model.encode(query_text)  # 384-dim vector
    
    # Semantic search in knowledge base
    knowledge_entries <- []
    FOR each entry IN KNOWLEDGE_BASE:
        entry_embedding <- get_cached_embedding(entry)
        similarity <- cosine_similarity(query_embedding, entry_embedding)
        IF similarity > threshold:
            knowledge_entries.append((entry, similarity))
    
    # Rank by semantic similarity
    ranked_entries <- sort(knowledge_entries, by=similarity, descending=True)
    top_knowledge <- select_top_k(ranked_entries, k=3)
    
    # Semantic PubMed retrieval
    pubmed_articles <- search_pubmed(class_name)
    FOR each article IN pubmed_articles:
        article.score <- cosine_similarity(query_embedding, embed(article.abstract))
    top_articles <- sort_by_score(pubmed_articles)[:2]
    
    # Compile medical context
    medical_context <- concatenate([r.content for r in top_knowledge])
    sources <- extract_citations(top_knowledge + top_articles)
    
    
    # ===============================================================
    # PHASE 6: EXPLANATION GENERATION
    # ===============================================================
    
    explanation <- format_explanation(
        prediction = class_name,
        confidence = confidence,
        visual_evidence = describe(visual_cues),
        medical_context = medical_context,
        sources = sources
    )
    
    
    # ===============================================================
    # OUTPUT
    # ===============================================================
    
    output <- {
        "prediction": class_name,
        "confidence": confidence,
        "probabilities": probabilities,
        "heatmap": heatmap,
        "overlay": blend(image, heatmap),
        "explanation": explanation
    }
    
    RETURN output

END FUNCTION
```

---

## Component Algorithms

### 1. MobileNetV2 with Transfer Learning

```
ALGORITHM: TransferLearning_MobileNetV2

# Load pretrained model (ImageNet weights)
base_model <- load_pretrained("MobileNetV2", weights="ImageNet")

# Modify classifier head
original_classifier <- base_model.classifier  # 1280 -> 1000 (ImageNet classes)
new_classifier <- Sequential(
    Dropout(p=0.2),
    Linear(1280 -> 5)  # 5 lung cancer classes
)
base_model.classifier <- new_classifier

# Training
FOR each epoch IN [1, ..., num_epochs]:
    FOR each batch IN training_data:
        outputs <- base_model(batch.images)
        loss <- CrossEntropyLoss(outputs, batch.labels)
        loss.backward()
        optimizer.step()  # AdamW, lr=1e-4
    
    val_accuracy <- evaluate(base_model, validation_data)
    IF val_accuracy > best_accuracy:
        save_checkpoint(base_model)
    IF no_improvement_for(patience=10):
        BREAK  # Early stopping
```

### 2. CNN Model Definitions (5 Comparison Models)

```
ALGORITHM: CNN_Model_Definitions

# ================== MODEL 1: MobileNetV2 (PRIMARY) ==================
mobilenetv2 <- load_pretrained("MobileNetV2", weights="ImageNet")
mobilenetv2.classifier <- Sequential(Dropout(0.2), Linear(1280 -> 5))
# Target layer for GradCAM: features[18]
# Parameters: 2.2M | Baseline: 89.61% | Fine-tuned: 97.40%

# ================== MODEL 2: DenseNet-121 ==================
densenet121 <- load_pretrained("DenseNet-121", weights="ImageNet")
densenet121.classifier <- Sequential(Dropout(0.5), Linear(1024 -> 5))
# Target layer for GradCAM: features.denseblock4
# Parameters: 7.0M | Baseline: 84.42%

# ================== MODEL 3: ResNet-50 ==================
resnet50 <- load_pretrained("ResNet-50", weights="ImageNet")
resnet50.fc <- Sequential(Dropout(0.5), Linear(2048 -> 5))
# Target layer for GradCAM: layer4
# Parameters: 23.5M | Baseline: 78.79% | Fine-tuned: 96.97%

# ================== MODEL 4: EfficientNet-B0 ==================
efficientnet_b0 <- load_pretrained("EfficientNet-B0", weights="ImageNet")
efficientnet_b0.classifier <- Sequential(Dropout(0.3), Linear(1280 -> 5))
# Target layer for GradCAM: features[8]
# Parameters: 5.3M | Baseline: 72.29%

# ================== MODEL 5: VGG-16 ==================
vgg16 <- load_pretrained("VGG-16", weights="ImageNet")
vgg16.classifier <- Sequential(
    Linear(25088 -> 4096), ReLU, Dropout(0.5),
    Linear(4096 -> 4096), ReLU, Dropout(0.5),
    Linear(4096 -> 5)
)
# Target layer for GradCAM: features[29]
# Parameters: 138M | Baseline: 71.43%
```

### 3. Multi-Model Training Pipeline

```
ALGORITHM: Train_All_CNN_Models

# Define all 5 CNN models for comparison
MODELS <- [
    ("mobilenetv2",    mobilenetv2,    1280, "features[18]"),      # PRIMARY
    ("densenet121",    densenet121,    1024, "features.denseblock4"),
    ("resnet50",       resnet50,       2048, "layer4"),
    ("efficientnet_b0", efficientnet_b0, 1280, "features[8]"),
    ("vgg16",          vgg16,          4096, "features[29]")
]

# Training loop for all models
FOR each (name, model, features, gradcam_layer) IN MODELS:
    print(f"Training {name}...")
    
    FOR each epoch IN [1, ..., num_epochs]:
        FOR each batch IN training_data:
            outputs <- model(batch.images)
            loss <- CrossEntropyLoss(outputs, batch.labels)
            loss.backward()
            optimizer.step()  # AdamW, lr=1e-4
        
        val_accuracy <- evaluate(model, validation_data)
        IF val_accuracy > best_accuracy[name]:
            save_checkpoint(model, f"best_model_{name}.pth")
        IF no_improvement_for(patience=10):
            BREAK  # Early stopping
    
    save_final(model, f"final_model_{name}.pth")
    results[name] <- evaluate(model, test_data)

# Compare results
print_comparison_table(results)
```

### 4. Grad-CAM Algorithm

```
ALGORITHM: GradCAM

INPUT:  model, input_image, target_class
OUTPUT: heatmap (same size as input)

# Step 1: Forward pass with hook
activations <- []
REGISTER forward_hook ON model.features[18]:  # MobileNetV2 last conv
    activations <- output

output <- model(input_image)

# Step 2: Backward pass
model.zero_grad()
target_score <- output[0, target_class]
target_score.backward()

gradients <- get_gradients(model.features[18])

# Step 3: Compute weights via GAP
weights <- mean(gradients, dim=[height, width])  # Shape: [channels]

# Step 4: Weighted combination
cam <- SUM(weights[c] * activations[c]) for c in channels

# Step 5: Apply ReLU (keep positive influences only)
cam <- ReLU(cam)

# Step 6: Normalize and resize
cam <- (cam - min(cam)) / (max(cam) - min(cam))
heatmap <- resize(cam, size=input_image.size)

RETURN heatmap
```

### 5. Semantic RAG Retrieval

```
ALGORITHM: SemanticRAG_Retrieval

INPUT:  predicted_class, visual_cues
OUTPUT: medical_context, sources

# Sentence Embedding Model: all-MiniLM-L6-v2
embedding_model <- load("all-MiniLM-L6-v2")  # 384-dim embeddings

# Knowledge Base (50+ curated entries with pre-computed embeddings)
KNOWLEDGE_BASE <- [
    {id: "adeno_001", 
     content: "Adenocarcinoma typically presents...",
     embedding: [0.12, -0.34, ...],  # 384-dim
     source: "WHO Classification 2021"},
    ...
]

# Step 1: Generate semantic query
query_text <- f"{predicted_class} {visual_cues.location} {visual_cues.pattern}"
query_embedding <- embedding_model.encode(query_text)  # 384-dim vector

# Step 2: Semantic search (cosine similarity)
results <- []
FOR each entry IN KNOWLEDGE_BASE:
    similarity <- cosine_similarity(query_embedding, entry.embedding)
    results.add((entry, similarity))

# Step 3: Rank and select
results <- sort(results, by=similarity, descending=True)
top_results <- results[0:3]

# Step 4: PubMed semantic retrieval
pubmed_articles <- search_pubmed(predicted_class)
FOR each article IN pubmed_articles:
    article.score <- cosine_similarity(query_embedding, 
                                       embedding_model.encode(article.abstract))
top_articles <- sort_by_score(pubmed_articles)[:2]

# Step 5: Format output
medical_context <- concatenate([r.content for r in top_results])
sources <- extract_citations(top_results + top_articles)

RETURN medical_context, sources
```

---

## Data Flow Summary

```
+------------------+
|   CT Scan        |
|   (Input)        |
+---------+--------+
          |
          v
+------------------+
|  Preprocessing   |  -> Resize to 224x224
|                  |  -> Normalize (ImageNet stats)
+---------+--------+
          |
          v
+------------------+
|  5 CNN MODELS    |
|                  |
|  MobileNetV2     |  -> 2.2M params  (PRIMARY: 89.61%)
|  DenseNet-121    |  -> 7.0M params  (Baseline: 84.42%)
|  ResNet-50       |  -> 23.5M params (Baseline: 78.79%)
|  EfficientNet-B0 |  -> 5.3M params  (Baseline: 72.29%)
|  VGG-16          |  -> 138M params  (Baseline: 71.43%)
+---------+--------+
          |
     +----+----+
     |         |
     v         v
+-------+ +----------+
|Predict| | Grad-CAM |
|Class  | | Heatmap  |
+---+---+ +----+-----+
    |          |
    |    +-----v-----+
    |    |  Analyze  |
    |    |  Regions  |
    |    +-----+-----+
    |          |
    |    +-----v--------+
    |    | Semantic RAG |<-- Knowledge Base
    |    | Retrieval    |    (50+ entries)
    |    +-----+--------+    + PubMed
    |          |
    +-----+----+
          |
          v
+------------------+
|   EXPLAINABLE    |
|     OUTPUT       |
|                  |
| * Prediction     |
| * Confidence     |
| * GradCAM Heatmap|
| * Medical Context|
| * Citations      |
+------------------+
```

---

## Class Definitions

| Class ID | Class Name              | Description                    |
|----------|------------------------|--------------------------------|
| 0        | Adenocarcinoma         | Most common lung cancer type   |
| 1        | Benign Cases           | Non-cancerous conditions       |
| 2        | Large Cell Carcinoma   | Aggressive cancer type         |
| 3        | Normal Cases           | Healthy lung tissue            |
| 4        | Squamous Cell Carcinoma| Central lung cancer            |

---

## Technical Specifications

### CNN Model Comparison (Baseline - Trained from Scratch)

| Model           | Parameters | Baseline Accuracy | Fine-tuned Accuracy | GradCAM Layer          |
|-----------------|------------|-------------------|---------------------|------------------------|
| **MobileNetV2** | 2.2M       | **89.61%**        | 97.40%              | features[18]           |
| DenseNet-121    | 7.0M       | 84.42%            | -                   | features.denseblock4   |
| ResNet-50       | 23.5M      | 78.79%            | 96.97%              | layer4                 |
| EfficientNet-B0 | 5.3M       | 72.29%            | -                   | features[8]            |
| VGG-16          | 138M       | 71.43%            | -                   | features[29]           |

**Note:** MobileNetV2 is the PRIMARY model - best baseline accuracy with smallest footprint.

### System Specifications

| Component          | Specification                          |
|--------------------|----------------------------------------|
| Primary Model      | MobileNetV2 (89.61% baseline)          |
| Total CNN Models   | 5 (comparative analysis)               |
| Input Size         | 224 x 224 x 3                          |
| XAI Method         | Grad-CAM (native CNN support)          |
| Knowledge Base     | 50+ curated medical entries            |
| Semantic Model     | all-MiniLM-L6-v2 (384-dim)            |
| Framework          | PyTorch 2.5.1 + CUDA 12.1             |
| GPU                | NVIDIA RTX 3060 (6GB VRAM)            |

---

## References

1. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR.
2. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
3. Huang, G., et al. (2017). "Densely Connected Convolutional Networks." CVPR.
4. Tan, M., and Le, Q.V. (2019). "EfficientNet: Rethinking Model Scaling." ICML.
5. Simonyan, K., and Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." ICLR.
6. Selvaraju, R.R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." ICCV.
7. Reimers, N. and Gurevych, I. (2019). "Sentence-BERT." EMNLP-IJCNLP.
8. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.

---

*Document prepared for Major Project Review*
*February 2026*
