# 🌿 Plant Disease Classifier - Production ML System

A **complete, production-ready deep learning system** for classifying plant diseases from images. 

This is not just code—it's a **teaching project** showing how to build professional ML systems that actually work in production.

---

## ⚡ Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Sample Data
```bash
python scripts/create_sample_data.py
```

### 3. Train the Model
```bash
python src/train.py
```

### 4. Start the API
```bash
python src/api.py
```

### 5. Make Predictions
Open browser: **http://localhost:8000/docs**

Click "Try it out" on `/predict` endpoint, upload an image, and get predictions!

---

## 📚 Documentation

Read these to understand the system:

| Document | Purpose | Time |
|----------|---------|------|
| **QUICKSTART.md** | Step-by-step guide | 10 min |
| **ARCHITECTURE.md** | How the system works | 20 min |
| **PRODUCTION_GUIDE.md** | Deep dive into each component | 30 min |

---

## 🎯 What You Get

### ✅ Complete System
- Data loading & preprocessing pipeline
- Transfer learning model training
- FastAPI web server for predictions
- Batch prediction support
- Health checks and monitoring
- Comprehensive test suite
- Docker containerization
- Production-grade logging

### ✅ Best Practices
- Configuration management (not hardcoded values)
- Separation of concerns (modular design)
- Error handling and validation
- Logging for debugging
- Unit tests for each component
- Documentation and comments

### ✅ Ready to Deploy
- Works locally
- Works in Docker
- Works on cloud (AWS, GCP, Azure)
- Can handle production traffic
- Includes monitoring

---

## 📁 Project Structure

```
plant-disease-classifier/
├── README.md                 ← You are here
├── QUICKSTART.md            ← Start here
├── ARCHITECTURE.md          ← System design
├── PRODUCTION_GUIDE.md      ← Deep dive
│
├── config.yaml              ← All settings (change to experiment)
├── requirements.txt         ← Python dependencies
├── Dockerfile              ← Container configuration
│
├── src/                     ← Source code
│   ├── train.py            ← Training entry point
│   ├── api.py              ← Web server
│   ├── data_pipeline.py    ← Load & preprocess data
│   ├── model_training.py   ← Training logic
│   ├── inference.py        ← Make predictions
│   └── utils.py            ← Logging, config, validation
│
├── tests/                  ← Test suite
│   └── test_pipeline.py    ← Comprehensive tests
│
├── scripts/                ← Utility scripts
│   └── create_sample_data.py ← Generate test images
│
├── data/                   ← Data directory
│   ├── raw/                ← Original images
│   │   ├── healthy/
│   │   ├── powdery_mildew/
│   │   ├── rust/
│   │   ├── leaf_spot/
│   │   └── blight/
│   └── processed/          ← (Optional) Processed images
│
├── models/                 ← Saved models
│   └── trained/
│       ├── plant_classifier_v1.h5
│       └── plant_classifier_v1_metadata.json
│
└── logs/                   ← Application logs
    ├── app.log
    └── errors.jsonl
```

---

## 🚀 Full Usage Guide

### Training a Model

```bash
# Generate sample data (optional, if you don't have real data)
python scripts/create_sample_data.py --num-samples 50

# Train the model
python src/train.py

# Output:
# - Model saved to: models/trained/plant_classifier_v1.h5
# - Metrics saved to: models/trained/plant_classifier_v1_metadata.json
# - Logs saved to: logs/app.log
```

### Running the API

```bash
# Start server
python src/api.py

# Server available at: http://localhost:8000

# Interactive docs: http://localhost:8000/docs
# Alternative docs: http://localhost:8000/redoc

# Health check: http://localhost:8000/health
# Model info: http://localhost:8000/model-info
```

### Making Predictions

#### Option 1: Web UI (Easiest)
1. Go to http://localhost:8000/docs
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Choose image file
5. Click "Execute"

#### Option 2: Command Line
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

#### Option 3: Python Script
```python
import requests

url = "http://localhost:8000/predict"
files = {'file': open('image.jpg', 'rb')}
response = requests.post(url, files=files)
result = response.json()

print(f"Disease: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Predictions

```bash
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_pipeline.py::TestDataLoader -v

# Run with coverage
pytest tests/ --cov=src
```

---

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t plant-classifier:latest .
```

### Run Container
```bash
docker run -p 8000:8000 plant-classifier:latest
```

### With GPU Support
```bash
docker run --gpus all -p 8000:8000 plant-classifier:latest
```

### Push to Registry
```bash
# AWS ECR
docker tag plant-classifier:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/plant-classifier:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/plant-classifier:latest

# Docker Hub
docker tag plant-classifier:latest yourusername/plant-classifier:latest
docker push yourusername/plant-classifier:latest
```

---

## 🔧 Configuration

All settings are in `config.yaml`. Change this file to experiment:

```yaml
# Image settings
data:
  image_size: 224  # Smaller = faster but less accurate
  batch_size: 32   # Larger = faster but needs more GPU RAM

# Model training
model:
  learning_rate: 1e-4   # Lower = more stable training
  epochs: 20            # More = longer training
  
# API server
api:
  port: 8000
  host: "0.0.0.0"
```

---

## 📊 Understanding the System

### Data Pipeline
```
Raw Images
    ↓
Load & Validate (data_pipeline.py)
    ↓
Normalize & Resize
    ↓
Split into train/val/test (stratified)
    ↓
Data Augmentation (training only)
    ↓
Ready for training
```

### Training Pipeline
```
Training Data
    ↓
Build Model (Transfer Learning with ResNet50)
    ↓
Train on Training Set
    ↓
Validate on Validation Set
    ↓
Early Stopping (stop when validation stops improving)
    ↓
Evaluate on Test Set
    ↓
Save Model & Metadata
```

### Inference Pipeline
```
User Request (Image Upload)
    ↓
Validate (File format, size, etc)
    ↓
Load & Preprocess Image
    ↓
Run Model Inference
    ↓
Format Response
    ↓
Return Prediction (JSON)
```

---

## 🎓 Learning Path

### Beginner
1. Run `python scripts/create_sample_data.py` 
2. Run `python src/train.py`
3. Run `python src/api.py`
4. Make predictions at http://localhost:8000/docs

### Intermediate
1. Read QUICKSTART.md
2. Modify `config.yaml` and retrain
3. Look at code comments (they explain reasoning)
4. Run tests: `pytest tests/ -v`

### Advanced
1. Read PRODUCTION_GUIDE.md
2. Read ARCHITECTURE.md
3. Study the code implementation
4. Extend with new features (authentication, monitoring, etc)

---

## 🛠️ Common Tasks

### I want to train on my own data
```
1. Organize images in data/raw/:
   data/raw/
   ├── healthy/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── disease/
       ├── image1.jpg
       └── ...

2. Update config.yaml classes
3. Run: python src/train.py
```

### I want to use a different model
```
1. Edit model_training.py ModelBuilder.build_model()
2. Change model architecture
3. Run: python src/train.py
```

### I want to add authentication
```
1. Install: pip install python-jose
2. Edit api.py
3. Add JWT token validation
```

### I want to monitor performance
```
1. Edit logging configuration in config.yaml
2. Setup monitoring service (Datadog, New Relic, etc)
3. Configure alerting
```

---

## 🐛 Troubleshooting

### Problem: "No module named 'tensorflow'"
**Solution**: `pip install -r requirements.txt`

### Problem: "No data found"
**Solution**: Run `python scripts/create_sample_data.py`

### Problem: "CUDA out of memory"
**Solution**: Reduce `batch_size` in config.yaml

### Problem: "Port 8000 in use"
**Solution**: Change `api.port` in config.yaml or kill process on port 8000

### Problem: "Model predictions are wrong"
**Solution**: 
- Need more training data
- Increase `epochs` in config.yaml
- Increase `rotation_range` for more augmentation
- Check that data preprocessing is correct

### Problem: "Tests failing"
**Solution**: 
1. Make sure you have test data
2. Run: `pip install pytest`
3. Run: `pytest tests/ -v` to see detailed errors

---

## 📈 Performance Benchmarks

On a typical machine:

| Operation | Time | Notes |
|-----------|------|-------|
| Data loading (1000 images) | 10-15s | Depends on disk speed |
| Model training (100 epochs) | 5-10 min | With GPU, CPU takes longer |
| Single prediction | 100-200ms | After model is loaded |
| Batch prediction (10 images) | 150-250ms | Much faster per image |

---

## 📦 Dependencies

All dependencies are in `requirements.txt`:

```
tensorflow==2.14.0      # Deep learning framework
fastapi==0.104.1        # Web API
uvicorn==0.24.0         # ASGI server
opencv-python==4.8.0    # Image processing
scikit-learn==1.3.0     # ML utilities
numpy==1.24.3           # Numerical computing
pydantic==2.4.2         # Data validation
pyyaml==6.0.1           # Config parsing
pytest==7.4.3           # Testing
```

---

## 🚀 Deployment Options

### Local Development
```bash
python src/api.py
```

### Docker (Recommended)
```bash
docker build -t plant-classifier:v1 .
docker run -p 8000:8000 plant-classifier:v1
```

### AWS (Elastic Beanstalk)
```bash
pip install awsebcli-ce
eb init
eb create plant-classifier-env
eb deploy
```

### Google Cloud (Cloud Run)
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/plant-classifier
gcloud run deploy --image gcr.io/PROJECT_ID/plant-classifier
```

### Azure (Container Instances)
```bash
az containerapp create --name plant-classifier \
  --resource-group myResourceGroup \
  --image plant-classifier:latest
```

---

## 📝 API Documentation

### Endpoints

#### `POST /predict`
Single image prediction.

**Request:**
```
Content-Type: multipart/form-data
Body: Binary image file (JPG, PNG)
```

**Response:**
```json
{
  "success": true,
  "class_name": "powdery_mildew",
  "confidence": 0.92,
  "is_confident": true,
  "all_predictions": {
    "healthy": 0.02,
    "powdery_mildew": 0.92,
    "rust": 0.03,
    "leaf_spot": 0.02,
    "blight": 0.01
  }
}
```

#### `POST /predict-batch`
Multiple image predictions.

**Request:**
```
Content-Type: multipart/form-data
Body: Multiple binary image files
```

**Response:**
```json
{
  "success": true,
  "predictions": [...],
  "summary": {
    "total": 3,
    "successful": 3,
    "failed": 0
  }
}
```

#### `GET /health`
Health check.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### `GET /model-info`
Get model information.

**Response:**
```json
{
  "model_type": "ResNet50 (Transfer Learning)",
  "num_classes": 5,
  "classes": ["healthy", "powdery_mildew", "rust", "leaf_spot", "blight"],
  "input_size": 224
}
```

---

## 🤝 Contributing

Found a bug? Want to improve something?

1. Make changes
2. Run tests: `pytest tests/ -v`
3. Check code quality
4. Create pull request

---

## 📄 License

This project is provided as-is for educational purposes.

---

## 🙋 Support

**Questions? Need help?**

1. Check QUICKSTART.md
2. Check PRODUCTION_GUIDE.md
3. Read code comments
4. Look at tests for usage examples

---

## 🎉 Next Steps

1. ✅ Run `python scripts/create_sample_data.py`
2. ✅ Run `python src/train.py`
3. ✅ Run `python src/api.py`
4. ✅ Open http://localhost:8000/docs
5. ✅ Read QUICKSTART.md
6. ✅ Experiment with config.yaml
7. ✅ Build on top of this system!

---

**You now have a production-ready ML system!** 🚀

Start with: `python scripts/create_sample_data.py`
