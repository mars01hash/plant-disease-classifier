# Quick Start Guide - Get Running in 10 Minutes

## Prerequisites
- Python 3.9+
- Git
- ~2GB disk space

## Step 1: Setup (2 minutes)

```bash
# Clone or download the project
cd plant-disease-classifier

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Data (1 minute)

Create this directory structure:
```
data/
└── raw/
    ├── healthy/
    │   ├── healthy_1.jpg
    │   ├── healthy_2.jpg
    │   └── ...
    ├── powdery_mildew/
    │   ├── mildew_1.jpg
    │   └── ...
    ├── rust/
    │   └── ...
    ├── leaf_spot/
    │   └── ...
    └── blight/
        └── ...
```

**Need test data?** Create dummy images for testing:
```python
# create_dummy_data.py
import numpy as np
from PIL import Image
from pathlib import Path

classes = ["healthy", "powdery_mildew", "rust", "leaf_spot", "blight"]

for class_name in classes:
    class_dir = Path(f"data/raw/{class_name}")
    class_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(10):  # 10 images per class for testing
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(img).save(f"{class_dir}/img_{i}.jpg")

print("Created dummy data!")
```

Run it:
```bash
python create_dummy_data.py
```

## Step 3: Train Model (3 minutes)

```bash
python src/train.py
```

**What you should see**:
```
======================================================================
PLANT DISEASE CLASSIFIER - TRAINING PIPELINE
======================================================================

[1/6] Loading configuration...
Configuration loaded successfully
Classes: ['healthy', 'powdery_mildew', 'rust', 'leaf_spot', 'blight']
Image size: 224x224

[2/6] Loading images from disk...
Loaded 50 images with shape (50, 224, 224, 3)

[3/6] Splitting data into train/val/test...
TRAIN: 28 images
VAL: 11 images
TEST: 11 images

[4/6] Building and training model...
Building transfer learning model
Model built successfully. Parameters: 23,589,637
Starting training for 20 epochs
Epoch 1/20 - loss: 1.4523 - val_loss: 1.3845
Epoch 2/20 - loss: 1.2341 - val_loss: 1.1234
... (continues for 20 epochs)

[5/6] Saving model and metadata...
Model saved to ./models/trained/plant_classifier_v1.h5

[6/6] Training complete!

TRAINING SUMMARY
============================================================
Key Metrics:
  - Training Accuracy: 0.8571
  - Validation Accuracy: 0.7273
  - Test Accuracy: 0.6364
  - Test Precision: 0.7200
  - Test Recall: 0.6364
  - Test F1 Score: 0.6765

Next steps:
1. Review metrics above
2. Start API server: python src/api.py
3. Test predictions at http://localhost:8000/docs
============================================================
```

## Step 4: Start API Server (1 minute)

```bash
python src/api.py
```

**You should see**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete
Model loaded successfully at startup
API is ready for requests!
```

## Step 5: Test Predictions (3 minutes)

### Option A: Interactive API Docs
Open browser: http://localhost:8000/docs

Click "Try it out" on `/predict` endpoint:
1. Click "Choose File"
2. Select a JPG or PNG image
3. Click "Execute"
4. See prediction!

### Option B: Command Line
```bash
# Test on a single image
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/raw/healthy/healthy_1.jpg"

# Response:
{
  "success": true,
  "class_name": "healthy",
  "confidence": 0.92,
  "is_confident": true,
  "all_predictions": {
    "healthy": 0.92,
    "powdery_mildew": 0.02,
    "rust": 0.03,
    "leaf_spot": 0.02,
    "blight": 0.01
  }
}
```

### Option C: Python Script
```python
import requests

url = "http://localhost:8000/predict"
files = {'file': open('data/raw/healthy/healthy_1.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

## Debugging

### Model didn't train well?
- **More data**: Collect more images
- **Longer training**: Increase `epochs` in config.yaml
- **Better augmentation**: Increase `rotation_range`, `zoom_range`
- **Lower learning rate**: Change `learning_rate` in config.yaml

### API not responding?
```bash
# Check health
curl http://localhost:8000/health

# Check logs
tail -f logs/app.log

# Restart server (stop with Ctrl+C, then):
python src/api.py
```

### Wrong predictions?
- Model might be underfitted (needs more training)
- Data might be poor quality
- Classes might be too similar
- Confidence threshold might be too high

### Out of memory?
- Reduce `batch_size` in config.yaml
- Use smaller `image_size` (e.g., 128 instead of 224)
- Close other applications

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src
```

## Deployment

### Docker
```bash
# Build
docker build -t plant-classifier:latest .

# Run
docker run -p 8000:8000 plant-classifier:latest

# With GPU support
docker run --gpus all -p 8000:8000 plant-classifier:latest
```

### Cloud Deployment

#### AWS (using Elastic Beanstalk)
```bash
pip install awsebcli-ce
eb init
eb create plant-classifier-env
eb deploy
```

#### Google Cloud (using Cloud Run)
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/plant-classifier
gcloud run deploy plant-classifier --image gcr.io/PROJECT_ID/plant-classifier
```

#### Azure (using App Service)
```bash
az containerapp create \
  --name plant-classifier \
  --resource-group myResourceGroup \
  --image plant-classifier:latest
```

## What's Next?

### Learn More
- Read `PRODUCTION_GUIDE.md` for deep dive
- Explore code comments (I explain reasoning)
- Modify config.yaml and retrain

### Improvements to Try
1. **Add authentication**: Protect API with API keys
2. **Add monitoring**: Track predictions and performance
3. **Add caching**: Speed up repeated predictions
4. **Try different models**: Use EfficientNet, ViT, or others
5. **Add explainability**: Show which parts of image triggered prediction

### Production Checklist
- [ ] Model trained and validated
- [ ] API tested locally
- [ ] Docker image builds successfully
- [ ] Tests pass (pytest)
- [ ] Logging works
- [ ] Health check responds
- [ ] Performance acceptable (<500ms per prediction)
- [ ] Deployed to production
- [ ] Monitoring/alerting configured
- [ ] Team can maintain/update

## Common Commands Reference

```bash
# Training
python src/train.py

# Start API
python src/api.py

# Run tests
pytest tests/ -v

# Check specific test
pytest tests/test_pipeline.py::TestDataSplitter -v

# Rebuild Docker image
docker build -t plant-classifier:latest .

# View logs during training
tail -f logs/app.log

# View model metadata
cat models/trained/plant_classifier_v1_metadata.json
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named 'tensorflow'" | `pip install tensorflow` |
| "No data found" | Create data/raw/ directory with class subdirectories |
| "CUDA out of memory" | Reduce batch_size or image_size in config.yaml |
| "Model not found" | Train model first: `python src/train.py` |
| "Port 8000 in use" | Change port in config.yaml or `lsof -i :8000` |
| "Predictions wrong" | Need more data or longer training |

## File Structure Quick Reference

```
plant-disease-classifier/
├── config.yaml           # ← CHANGE HERE for hyperparameters
├── requirements.txt      # ← Dependencies
├── src/
│   ├── train.py         # ← Run to train
│   ├── api.py           # ← Run to serve
│   ├── data_pipeline.py # ← Data loading
│   ├── model_training.py # ← Training logic
│   ├── inference.py     # ← Prediction logic
│   └── utils.py         # ← Helpers
├── data/
│   └── raw/             # ← Put images here
└── models/
    └── trained/         # ← Saved models go here
```

## Next Steps

1. **Read**: `PRODUCTION_GUIDE.md`
2. **Experiment**: Modify `config.yaml` and retrain
3. **Deploy**: Use Docker to run in production
4. **Monitor**: Add logging/alerting
5. **Maintain**: Retrain when performance degrades

---

**You now have a production-ready ML system! 🚀**

Questions? Check `PRODUCTION_GUIDE.md` or look at code comments.
