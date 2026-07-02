# Production ML Application Guide
## Plant Disease Classifier - Complete Tutorial

This guide teaches you how to build production-ready ML systems from scratch.

---

## Table of Contents
1. [My Mental Model](#my-mental-model)
2. [System Architecture](#system-architecture)
3. [How to Use This](#how-to-use-this)
4. [Deep Dive: Each Component](#deep-dive)
5. [Deployment](#deployment)
6. [Monitoring & Maintenance](#monitoring)
7. [Common Mistakes & Solutions](#common-mistakes)

---

## My Mental Model

When I build production ML systems, I think in these layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    USER REQUESTS (API)                      │
│                   (FastAPI Server)                          │
├─────────────────────────────────────────────────────────────┤
│                    REQUEST VALIDATION                       │
│           (Check format, size, authentication)              │
├─────────────────────────────────────────────────────────────┤
│                     MODEL INFERENCE                         │
│              (Make predictions on clean data)               │
├─────────────────────────────────────────────────────────────┤
│                   PRE-PROCESSING                            │
│         (Prepare data in exact same way as training)        │
├─────────────────────────────────────────────────────────────┤
│                     TRAINED MODEL                           │
│           (Neural network with learned weights)             │
├─────────────────────────────────────────────────────────────┤
│              CONFIGURATION & LOGGING                        │
│    (Settings, tracking, monitoring, debugging)             │
├─────────────────────────────────────────────────────────────┤
│                    DATA PIPELINE                            │
│      (Loading, validation, splitting, augmentation)        │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: Each layer solves one problem and doesn't depend on implementation details of other layers.

---

## System Architecture

### Components Explained

#### 1. Configuration (`config.yaml`)
**Purpose**: Single source of truth for all settings

```yaml
# ✓ Good: Change settings without touching code
learning_rate: 1e-4

# ✗ Bad: Hardcoded values scattered through code
model.lr = 0.0001
```

**Why matter**: 
- Change hyperparameters without retraining
- Different configs for dev/test/prod
- Reproducibility (document exact settings used)

---

#### 2. Data Pipeline (`data_pipeline.py`)

**My philosophy**: Data is the foundation. Spend 80% of time here, 20% on models.

**Components**:

##### a. DataLoader
```python
# Responsibility: Load images from disk efficiently
loader = ImageDataLoader(config, logger)
images, labels, paths = loader.load_images("data/raw")

# Why separate:
# - Handles different formats (JPG, PNG, etc)
# - Validates images as they load
# - Returns file paths for debugging
# - Consistent preprocessing for all images
```

**Critical**: Preprocessing must be IDENTICAL during training and inference.
```
During Training:
image → load → resize to 224x224 → normalize [0,1] → train

During Inference:
image → load → resize to 224x224 → normalize [0,1] → predict

If different: model performs poorly in production! ❌
```

##### b. DataAugmentor
```python
# Augmentation prevents overfitting
# Use ONLY during training, never during validation/test

# Example: rotate image by random angle
# Why: Model becomes invariant to rotation variations
# Real-world: Farmers photograph plants at different angles
```

##### c. DataSplitter
```python
# Split into train/val/test with stratification
# Stratification = maintain class distribution in each split

# Example: If 80% healthy, 20% disease in original:
# - Train set: 80% healthy, 20% disease ✓
# - Val set: 80% healthy, 20% disease ✓
# - Test set: 80% healthy, 20% disease ✓

# NOT stratified would give wrong performance estimate!
```

**Key Rule**:
- Never touch test set during training/tuning
- Use validation set to pick best model
- Use test set only for final evaluation

---

#### 3. Model Training (`model_training.py`)

**Transfer Learning Approach**:

```
Pre-trained ResNet50 (trained on 1M images from ImageNet)
        ↓
Knows how to detect: edges, corners, textures, shapes
        ↓
Replace final layer (our 5 diseases)
        ↓
Train only NEW layer (keep pre-trained weights frozen)
        ↓
Benefits:
- Learns much faster (fewer parameters to train)
- Needs less data (pre-trained weights already good)
- Better generalization
```

**Why better than training from scratch**:
```
Training from scratch:
- Need 100k+ images
- Takes weeks on GPU
- Often overfits with small dataset
- Poor performance in production

Transfer learning:
- Works with 1k images
- Trains in hours
- Pre-trained weights prevent overfitting
- Better production performance
```

**How to train**:
```python
from model_training import train_full_pipeline

results = train_full_pipeline(
    config, logger,
    train_images, train_labels,
    val_images, val_labels,
    test_images, test_labels
)
```

---

#### 4. Inference (`inference.py`)

**Three classes with different purposes**:

##### ModelLoader
```python
# Load model ONCE at startup, cache it
# Why: Loading takes 1-2 seconds
# If you load in every prediction request = 100 requests = 100-200 seconds waste!

loader = ModelLoader(logger)
model = loader.load_model("models/plant_classifier.h5")  # Loads once
model = loader.load_model("models/plant_classifier.h5")  # Returns cached version
```

##### ImagePredictor
```python
# Make single predictions
predictor = ImagePredictor(model, config, logger, classes)
result = predictor.predict_from_file("leaf.jpg")

# Returns:
{
    'prediction': {
        'class': 'powdery_mildew',
        'confidence': 0.92,
        'is_confident': True
    },
    'all_predictions': {
        'healthy': 0.02,
        'powdery_mildew': 0.92,
        ...
    }
}
```

##### BatchPredictor
```python
# Make predictions on multiple images at once
# MUCH faster than individual predictions (GPU utilization)

batch_pred = BatchPredictor(model, config, logger, classes)
results = batch_pred.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

# Why faster:
# - GPU loves batch processing
# - Less overhead per image
# - Can process 100 images while taking ~10x single prediction time
```

---

#### 5. API Server (`api.py`)

**FastAPI provides**:
- Automatic validation (Pydantic models)
- Interactive documentation (Swagger UI)
- Async/concurrent request handling
- Type hints for IDE support

**Key endpoints**:

```python
# Single prediction
POST /predict
- Input: Image file (JPG/PNG)
- Output: Prediction with confidence

# Batch prediction
POST /predict-batch
- Input: Multiple images
- Output: Predictions for all

# Health check
GET /health
- Used by load balancers to know if service is alive
- Allows graceful shutdown

# Model info
GET /model-info
- What classes can it predict?
- What's the model type?
- What version is running?
```

---

## How to Use This

### Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data in directory structure
data/raw/
├── healthy/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── powdery_mildew/
│   ├── img1.jpg
│   └── ...
└── (other diseases)
```

### Training
```bash
# Run full pipeline
python src/train.py

# What happens:
# 1. Loads config.yaml
# 2. Loads images from data/raw
# 3. Validates images
# 4. Splits into train/val/test
# 5. Trains model (uses GPU if available)
# 6. Evaluates on test set
# 7. Saves model + metadata
# 8. Prints summary with metrics
```

### Testing
```bash
# Run test suite
pytest tests/ -v

# What gets tested:
# - Data loading and validation
# - Train/val/test splitting
# - Configuration
# - Inference speed
# - Integration (full pipeline)
```

### Inference
```bash
# Start API server
python src/api.py

# Server starts at http://localhost:8000

# Try in browser:
# http://localhost:8000/docs (interactive API docs)

# Example prediction:
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@leaf.jpg"
```

### Deployment
```bash
# Build Docker image
docker build -t plant-classifier:latest .

# Run container
docker run -p 8000:8000 plant-classifier:latest

# With GPU:
docker run --gpus all -p 8000:8000 plant-classifier:latest
```

---

## Deep Dive: Each Component

### Component 1: Data Pipeline

**Why get this right**:
- Bad data → bad model (no amount of tuning fixes it)
- Good data → decent model (even simple models work)

**Common mistakes & fixes**:

```python
# ✗ WRONG: Different preprocessing
# Training
image = cv2.resize(image, (224, 224))
image = image / 255.0  # Normalize to [0,1]

# Inference
image = cv2.resize(image, (256, 256))  # ← Different size!
image = image / 255.0

# Result: Bad performance in production ❌


# ✓ RIGHT: Consistent preprocessing
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return image

# Use same function everywhere
train_image = preprocess_image(train_image)
test_image = preprocess_image(test_image)
```

**Augmentation strategy**:
```python
# USE ONLY during training
train_generator = DataGenerator(
    train_images, train_labels,
    augment=True,  # ← Only for training!
    augmentor=augmentor
)

# NOT during validation/test
val_generator = DataGenerator(
    val_images, val_labels,
    augment=False  # ← Never augment validation/test!
)

# Why:
# Training: Augmentation prevents overfitting
# Validation: Augmentation distorts evaluation metrics
# Test: Test set should reflect real-world data exactly
```

---

### Component 2: Model Training

**Transfer learning workflow**:

```
Step 1: Load pre-trained model
├─ ResNet50 from ImageNet
├─ Already knows visual features
└─ 50 layers deep

Step 2: Freeze backbone
├─ Don't update pre-trained weights
├─ Only train new classification layer
└─ Much faster + less data needed

Step 3: Add custom layers
├─ Global Average Pooling (reduce to 1D)
├─ Dropout (prevent overfitting)
└─ Dense layer (output predictions)

Step 4: Train only new layers
├─ Learning rate: 1e-4 (small, be careful)
├─ Loss: categorical_crossentropy
└─ Metrics: accuracy, precision, recall

Step 5: Early stopping
├─ Stop when validation stops improving
├─ Prevents overfitting
└─ Saves best model automatically
```

**Monitoring training**:

```python
# During training, watch these metrics:

# Training Loss going down ✓ = learning
# Training Loss plateauing ✓ = converged

# Validation Loss going down ✓ = generalizing
# Validation Loss going up ✗ = overfitting!

# If overfitting:
# 1. Add more augmentation
# 2. Increase dropout
# 3. Use more data
# 4. Reduce model complexity
```

---

### Component 3: Inference & Serving

**Performance optimization**:

```python
# ✗ SLOW: Load model for every request
@app.post("/predict")
async def predict(file):
    model = load_model("model.h5")  # ← 1 second per request!
    result = model.predict(image)
    return result

# ✓ FAST: Load once at startup
loaded_model = None

@app.on_event("startup")
async def startup():
    global loaded_model
    loaded_model = load_model("model.h5")  # ← Once at startup

@app.post("/predict")
async def predict(file):
    result = loaded_model.predict(image)  # ← 100ms total
    return result
```

**Error handling**:

```python
# Always validate input before processing
def predict(file):
    # Check file exists
    if file is None:
        raise HTTPException(400, "No file provided")
    
    # Check file size
    if len(file) > 5MB:
        raise HTTPException(413, "File too large")
    
    # Check file format
    if not file.endswith(('.jpg', '.png')):
        raise HTTPException(400, "Invalid format")
    
    # Now safe to process
    return model.predict(file)
```

---

## Deployment

### Local Development
```bash
python src/api.py
# Runs on http://localhost:8000
# Non-threaded, not production-ready
```

### Production Deployment

```bash
# Option 1: Docker (recommended)
docker run -p 8000:8000 plant-classifier:latest

# Option 2: Kubernetes
kubectl apply -f deployment.yaml

# Option 3: Cloud (AWS/GCP/Azure)
# Push image to registry
# Deploy to cloud platform (App Engine, Lambda, etc)

# Option 4: Gunicorn + Uvicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api:app
```

### Environment Variables
```bash
# Override config with environment variables
export MODEL_PATH="/models/v2/classifier.h5"
export API_PORT=8080
export LOG_LEVEL=DEBUG

# Useful for:
# - Different models in different environments
# - Secrets (API keys, database URLs)
# - Performance tuning per hardware
```

---

## Monitoring & Maintenance

### Health Checks
```python
# Kubernetes/load balancers use health checks
GET /health

# Should return:
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Logging
```python
# Every request is logged
logger.info(f"Prediction: {class} ({confidence:.2%})")

# Log file location: ./logs/app.log
# Use for:
# - Debugging errors
# - Understanding user patterns
# - Performance analysis
```

### Metrics to Track
```
1. Prediction latency
   - P50: median time
   - P95: 95th percentile
   - P99: 99th percentile

2. Error rate
   - Failed predictions
   - Invalid inputs
   - System errors

3. Model performance
   - Accuracy on recent data
   - Confidence distribution
   - False positive rate

4. System health
   - GPU memory usage
   - CPU usage
   - Request throughput
```

### Model Retraining
```
When to retrain:
1. New data collected (quarterly)
2. Performance degradation detected
3. New classes needed
4. Better hyperparameters found

How to retrain:
1. Run training pipeline on new data
2. Evaluate on test set
3. Compare metrics to current model
4. If better: deploy new version
5. If worse: investigate why

Always keep versioning:
- models/v1.0/classifier.h5
- models/v1.1/classifier.h5
- models/v2.0/classifier.h5

Can rollback if new version fails!
```

---

## Common Mistakes & Solutions

### Mistake 1: Different Preprocessing in Training vs Inference
```python
# ✗ This kills production performance
# training.py
image = cv2.resize(image, (224, 224))

# api.py
image = cv2.resize(image, (256, 256))

# ✓ Fix: Use same preprocessing function everywhere
def preprocess(image):
    return cv2.resize(image, (224, 224))
```

### Mistake 2: Loading Model for Every Request
```python
# ✗ Slow
@app.post("/predict")
async def predict(file):
    model = keras.models.load_model("model.h5")  # 1-2 seconds!

# ✓ Fast
@app.on_event("startup")
async def startup():
    global model
    model = keras.models.load_model("model.h5")
```

### Mistake 3: Trusting User Input
```python
# ✗ Will crash
def predict(file):
    image = cv2.imread(file)
    return model.predict(image)

# ✓ Robust
def predict(file):
    if not Path(file).exists():
        raise HTTPException(404, "File not found")
    if Path(file).stat().st_size > MAX_SIZE:
        raise HTTPException(413, "File too large")
    image = cv2.imread(file)
    if image is None:
        raise HTTPException(400, "Invalid image")
    return model.predict(image)
```

### Mistake 4: Not Testing
```python
# ✗ No tests = bugs in production

# ✓ Write tests
pytest tests/ -v

# Test:
# - Data loading
# - Preprocessing
# - Model inference
# - API endpoints
# - Error handling
```

### Mistake 5: Hardcoding Hyperparameters
```python
# ✗ Hard to experiment
model = keras.Sequential([
    layers.Dense(128),
    layers.Dense(64),
    layers.Dense(32)
])

# ✓ Use config
layers = config["model"]["layers"]  # [128, 64, 32]
model = keras.Sequential([
    layers.Dense(n) for n in layers
])
```

---

## Next Steps

### To Build Similar Systems:

1. **Start with architecture**: Diagram the data flow before coding
2. **Separate concerns**: Each component does one thing
3. **Config first**: Put all settings in one file
4. **Test early**: Write tests as you code
5. **Log everything**: Debug production issues using logs
6. **Version control**: Track model versions separately from code
7. **Monitor always**: Know when things break
8. **Document**: Future you will thank you

### To Improve This System:

1. Add authentication (API key, JWT)
2. Add rate limiting (prevent abuse)
3. Add caching (avoid redundant predictions)
4. Add A/B testing (test model versions)
5. Add monitoring/alerting (know when metrics degrade)
6. Add load balancing (handle spike in traffic)
7. Add model versioning (rollback if new version fails)

---

## Resources

- **TensorFlow/Keras**: https://www.tensorflow.org/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Docker**: https://www.docker.com/
- **Scikit-learn**: https://scikit-learn.org/
- **MLOps Best Practices**: https://ml-ops.systems/

---

## Summary

**Key Principles for Production ML**:

1. **Separation of Concerns**: Each component has one responsibility
2. **Reproducibility**: Same config + same data = same results
3. **Robustness**: Handle errors gracefully, validate everything
4. **Observability**: Log, monitor, track metrics
5. **Testability**: Test each component independently
6. **Simplicity**: Simple > clever (easier to debug, maintain)

Start here, and you can build scalable ML systems!
