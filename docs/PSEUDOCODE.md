# Explainable AI for Lung Cancer Classification
## Project Pseudocode & Algorithm

---

## 🎯 Main Pipeline Algorithm

```
ALGORITHM: ExplainableLungCancerClassification

INPUT:  CT scan image (chest X-ray or CT slice)
OUTPUT: Prediction with visual and textual explanation

FUNCTION lung_cancer_pipeline(input_image):
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: IMAGE PREPROCESSING
    # ═══════════════════════════════════════════════════════════════
    
    image ← load_image(input_image)
    image ← resize(image, size=224×224)
    image ← normalize(image, mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
    tensor ← convert_to_tensor(image)
    
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: MODEL PREDICTION
    # ═══════════════════════════════════════════════════════════════
    
    logits ← ResNet50_Model(tensor)
    probabilities ← softmax(logits)
    prediction ← argmax(probabilities)
    confidence ← max(probabilities)
    
    class_name ← CLASS_NAMES[prediction]
    # CLASS_NAMES = [adenocarcinoma, benign, large_cell, normal, squamous]
    
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: GRAD-CAM VISUALIZATION (XAI)
    # ═══════════════════════════════════════════════════════════════
    
    # Get activations from last convolutional layer
    activations ← forward_hook(model.layer4, tensor)
    
    # Compute gradients via backpropagation
    target_score ← logits[prediction]
    gradients ← backward(target_score)
    
    # Compute importance weights (Global Average Pooling)
    weights ← global_average_pool(gradients)
    
    # Generate weighted activation map
    heatmap ← ReLU(Σ weights × activations)
    heatmap ← normalize(heatmap, range=[0, 1])
    heatmap ← resize(heatmap, size=224×224)
    
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: HEATMAP ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    
    # Analyze spatial distribution
    attention_regions ← find_high_attention_regions(heatmap, threshold=0.5)
    spatial_location ← classify_location(attention_regions)
    # locations: upper_left, upper_right, lower_left, lower_right, central
    
    # Analyze intensity
    intensity_score ← mean(heatmap[attention_regions])
    coverage ← count(heatmap > 0.5) / total_pixels
    
    # Generate visual description
    visual_cues ← {
        "location": spatial_location,
        "intensity": intensity_score,
        "coverage": coverage,
        "pattern": detect_pattern(attention_regions)
    }
    
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: RAG-BASED KNOWLEDGE RETRIEVAL
    # ═══════════════════════════════════════════════════════════════
    
    # Generate search keywords
    keywords ← extract_keywords(class_name, visual_cues)
    # Example: ["adenocarcinoma", "peripheral", "ground glass"]
    
    # Retrieve relevant medical knowledge
    knowledge_entries ← []
    FOR each keyword IN keywords:
        matches ← search_knowledge_base(keyword)
        knowledge_entries.append(matches)
    
    # Rank by relevance
    ranked_entries ← rank_by_relevance(knowledge_entries, keywords)
    top_knowledge ← select_top_k(ranked_entries, k=3)
    
    # Compile medical context
    medical_context ← concatenate(top_knowledge)
    sources ← extract_citations(top_knowledge)
    
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 6: EXPLANATION GENERATION
    # ═══════════════════════════════════════════════════════════════
    
    explanation ← format_explanation(
        prediction = class_name,
        confidence = confidence,
        visual_evidence = describe(visual_cues),
        medical_context = medical_context,
        sources = sources
    )
    
    
    # ═══════════════════════════════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════════════════════════════
    
    output ← {
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

## 📊 Component Algorithms

### 1. ResNet-50 with Transfer Learning

```
ALGORITHM: TransferLearning_ResNet50

# Load pretrained model (ImageNet weights)
base_model ← load_pretrained("ResNet-50", weights="ImageNet")

# Modify classifier head
original_fc ← base_model.fc  # 2048 → 1000 (ImageNet classes)
new_fc ← Sequential(
    Dropout(p=0.5),
    Linear(2048 → 5)  # 5 lung cancer classes
)
base_model.fc ← new_fc

# Training
FOR each epoch IN [1, ..., num_epochs]:
    FOR each batch IN training_data:
        outputs ← base_model(batch.images)
        loss ← CrossEntropyLoss(outputs, batch.labels)
        loss.backward()
        optimizer.step()
    
    val_accuracy ← evaluate(base_model, validation_data)
    IF val_accuracy > best_accuracy:
        save_checkpoint(base_model)
```

### 2. Grad-CAM Algorithm

```
ALGORITHM: GradCAM

INPUT:  model, input_image, target_class
OUTPUT: heatmap (same size as input)

# Step 1: Forward pass with hook
activations ← []
REGISTER forward_hook ON model.layer4:
    activations ← output

output ← model(input_image)

# Step 2: Backward pass
model.zero_grad()
target_score ← output[0, target_class]
target_score.backward()

gradients ← get_gradients(model.layer4)

# Step 3: Compute weights via GAP
weights ← mean(gradients, dim=[height, width])  # Shape: [channels]

# Step 4: Weighted combination
cam ← Σ(weights[c] × activations[c]) for c in channels

# Step 5: Apply ReLU (keep positive influences only)
cam ← ReLU(cam)

# Step 6: Normalize and resize
cam ← (cam - min(cam)) / (max(cam) - min(cam))
heatmap ← resize(cam, size=input_image.size)

RETURN heatmap
```

### 3. RAG Knowledge Retrieval

```
ALGORITHM: RAG_Retrieval

INPUT:  predicted_class, visual_cues
OUTPUT: medical_context, sources

# Knowledge Base Structure
KNOWLEDGE_BASE ← [
    {id: "adeno_001", 
     keywords: ["adenocarcinoma", "peripheral"],
     content: "Adenocarcinoma typically presents...",
     source: "WHO Classification 2021"},
    ...
]

# Step 1: Generate query keywords
keywords ← []
keywords.add(predicted_class)
keywords.add(visual_cues.location_keywords)
keywords.add(visual_cues.pattern_keywords)

# Step 2: Search knowledge base
results ← []
FOR each entry IN KNOWLEDGE_BASE:
    score ← 0
    FOR each keyword IN keywords:
        IF keyword IN entry.keywords:
            score ← score + 1
    IF score > 0:
        results.add((entry, score))

# Step 3: Rank and select
results ← sort(results, by=score, descending=True)
top_results ← results[0:3]

# Step 4: Format output
medical_context ← concatenate([r.content for r in top_results])
sources ← [r.source for r in top_results]

RETURN medical_context, sources
```

---

## 🔄 Data Flow Summary

```
┌─────────────────┐
│   CT Scan       │
│   (Input)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  → Resize to 224×224
│                 │  → Normalize (ImageNet stats)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ResNet-50     │  → 23.5M parameters
│   (Backbone)    │  → Pretrained on ImageNet
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────────┐
│Predict│ │  Grad-CAM │
│Class  │ │  Heatmap  │
└───┬───┘ └─────┬─────┘
    │           │
    │     ┌─────▼─────┐
    │     │  Analyze  │
    │     │  Regions  │
    │     └─────┬─────┘
    │           │
    │     ┌─────▼─────┐
    │     │    RAG    │◄── Knowledge Base
    │     │ Retrieval │    (19 entries)
    │     └─────┬─────┘
    │           │
    └─────┬─────┘
          │
          ▼
┌─────────────────┐
│   EXPLAINABLE   │
│     OUTPUT      │
│                 │
│ • Prediction    │
│ • Confidence    │
│ • Heatmap       │
│ • Explanation   │
└─────────────────┘
```

---

## 📋 Class Definitions

| Class ID | Class Name              | Description                    |
|----------|------------------------|--------------------------------|
| 0        | Adenocarcinoma         | Most common lung cancer type   |
| 1        | Benign Cases           | Non-cancerous conditions       |
| 2        | Large Cell Carcinoma   | Aggressive cancer type         |
| 3        | Normal Cases           | Healthy lung tissue            |
| 4        | Squamous Cell Carcinoma| Central lung cancer            |

---

## 🛠️ Technical Specifications

| Component          | Specification                          |
|--------------------|----------------------------------------|
| Model Architecture | ResNet-50 (Modified)                   |
| Input Size         | 224 × 224 × 3                         |
| Parameters         | 23,518,277                             |
| XAI Method         | Grad-CAM (layer4)                      |
| Knowledge Base     | 19 curated medical entries             |
| Framework          | PyTorch 2.5.1 + CUDA 12.1             |
| GPU                | NVIDIA RTX 3060 (6GB VRAM)            |

---

## 📚 References

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
2. Selvaraju, R.R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." ICCV.
3. Travis, W.D., et al. (2021). "WHO Classification of Tumours of the Lung."
4. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."

---

*Document prepared for Major Project Review*
*December 2025*
