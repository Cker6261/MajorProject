# LungXAI - Command Reference Guide

A complete list of terminal commands for running the LungXAI project.

---

## 1. Environment Setup

### Create Virtual Environment
```powershell
python -m venv .venv
```

### Activate Virtual Environment (Windows PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

### Activate Virtual Environment (Windows CMD)
```cmd
.venv\Scripts\activate.bat
```

### Activate Virtual Environment (Linux/Mac)
```bash
source .venv/bin/activate
```

### Install Dependencies
```powershell
pip install -r requirements.txt
```

### Verify Installation
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 2. Training Models

### Train All Models (with caching)
```powershell
python train_all_models.py
```

### Force Retrain All Models
```powershell
python train_all_models.py --force-retrain
```

### Train Specific Models Only
```powershell
python train_all_models.py --models resnet50 mobilenetv2
python train_all_models.py --models vit_b_16 swin_t
```

### Train Single Model (Legacy)
```powershell
python main.py --mode train --epochs 30
```

### Train with Custom Parameters
```powershell
python main.py --mode train --epochs 50 --batch-size 16 --lr 0.0001
```

---

## 3. Model Comparison

### Compare All Trained Models
```powershell
python compare_models.py
```

This generates:
- `results/comparison/model_comparison_charts.png`
- `results/comparison/model_comparison_radar.png`
- `results/comparison/confusion_matrices_comparison.png`
- `results/comparison/model_comparison_report.md`

---

## 4. Demo & Prediction

### Demo with Default Model (ResNet-50)
```powershell
python demo_multi_model.py
```

### Demo with Specific Models
```powershell
python demo_multi_model.py --model resnet50
python demo_multi_model.py --model mobilenetv2
python demo_multi_model.py --model vit_b_16
python demo_multi_model.py --model swin_t
```

### Demo with Specific Image
```powershell
python demo_multi_model.py --model mobilenetv2 --image "path/to/your/image.png"
```

### Compare All Models on Same Image
```powershell
python demo_multi_model.py --compare
```

### List Available Models and Status
```powershell
python demo_multi_model.py --list
```

### Visual Demo (Simple)
```powershell
python demo.py
python demo.py "path/to/ct_scan.png"
```

---

## 5. Evaluation

### Evaluate Model on Test Set
```powershell
python main.py --mode evaluate --checkpoint checkpoints/best_model_resnet50.pth
python main.py --mode evaluate --checkpoint checkpoints/best_model_mobilenetv2.pth
```

### Evaluate Specific Model
```powershell
python evaluate_model.py --model resnet50
python evaluate_model.py --model mobilenetv2
python evaluate_model.py --model vit_b_16
python evaluate_model.py --model swin_t
```

---

## 6. Single Image Prediction

### Predict Single Image
```powershell
python main.py --mode predict --image "path/to/ct_scan.png"
```

### Predict with Specific Model
```powershell
python main.py --mode predict --image "path/to/ct_scan.png" --model mobilenetv2
```

---

## 7. Testing RAG Pipeline

### Test Complete RAG Pipeline
```powershell
python test_rag_pipeline.py
```

This tests:
- Knowledge Base (50 entries)
- PubMed Integration
- XAI to Text Conversion
- Explanation Generation
- Full Pipeline Integration

---

## 8. Jupyter Notebooks

### Start Jupyter Notebook
```powershell
jupyter notebook
```

### Start Jupyter Lab
```powershell
jupyter lab
```

### Run Specific Notebook
```powershell
jupyter notebook notebooks/LungXAI_Complete_Pipeline.ipynb
```

---

## 9. Utility Commands

### Check Python Version
```powershell
python --version
```

### Check GPU Status
```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### List Installed Packages
```powershell
pip list
```

### Check Project Structure
```powershell
Get-ChildItem -Recurse -Depth 2 | Where-Object { $_.Name -notmatch '__pycache__|\.git|\.venv|cache' }
```

### Clear Python Cache
```powershell
Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
```

---

## 10. Model Checkpoints

Available checkpoints in `checkpoints/` folder:
- `best_model_resnet50.pth` - ResNet-50 (96.97% accuracy)
- `best_model_mobilenetv2.pth` - MobileNetV2 (97.40% accuracy)
- `best_model_vit_b_16.pth` - ViT-B/16 (93.51% accuracy)
- `best_model_swin_t.pth` - Swin-T (97.84% accuracy - **Best**)

---

## 11. Quick Start (Full Workflow)

```powershell
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Train all models (skip if already trained)
python train_all_models.py

# 3. Compare models
python compare_models.py

# 4. Run demo with best model (Swin-T)
python demo_multi_model.py --model swin_t

# 5. Test RAG pipeline
python test_rag_pipeline.py
```

---

## 12. Troubleshooting

### If CUDA Out of Memory
```powershell
# Use smaller batch size
python train_all_models.py --batch-size 8

# Or train models one at a time
python train_all_models.py --models mobilenetv2
python train_all_models.py --models resnet50
```

### If Package Import Error
```powershell
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### If Virtual Environment Not Activating
```powershell
# Check execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try again
.\.venv\Scripts\Activate.ps1
```

### Check Current Python Executable
```powershell
python -c "import sys; print(sys.executable)"
```

---

## Summary Table

| Task | Command |
|------|---------|
| Activate venv | `.\.venv\Scripts\Activate.ps1` |
| Train all models | `python train_all_models.py` |
| Compare models | `python compare_models.py` |
| Demo (default) | `python demo_multi_model.py` |
| Demo (MobileNetV2) | `python demo_multi_model.py --model mobilenetv2` |
| Demo (Swin-T) | `python demo_multi_model.py --model swin_t` |
| Test RAG | `python test_rag_pipeline.py` |
| Evaluate model | `python evaluate_model.py --model resnet50` |
| Predict image | `python main.py --mode predict --image "path.png"` |

---

**Last Updated:** January 2026
