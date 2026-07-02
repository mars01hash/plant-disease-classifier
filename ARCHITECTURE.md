# Production ML System Architecture

## Overview

This is a **complete, production-ready ML system** for image classification. It shows my thinking process and all patterns used in real companies.

---

## My Decision-Making Framework

When I build production systems, I ask these questions:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. What problem am I solving?                              │
│    → Image classification for plant disease detection      │
│                                                             │
│ 2. What's the data flow?                                   │
│    → User uploads image → API validates → Model predicts  │
│                                                             │
│ 3. What can break?                                         │
│    → Bad data, network errors, invalid images, etc.       │
│                                                             │
│ 4. How do I make each piece testable?                      │
│    → Separate concerns, dependency injection              │
│                                                             │
│ 5. How do I monitor in production?                         │
│    → Logging, metrics, health checks                       │
│                                                             │
│ 6. How do I make it reproducible?                          │
│    → Config files, version control, documentation         │
└─────────────────────────────────────────────────────────────┘
```

---

## Complete File Structure

```
plant-disease-classifier/
│
├── 📋 CONFIGURATION & DOCUMENTATION
│   ├── config.yaml              ← All settings in ONE place
│   ├── requirements.txt         ← Python dependencies
│   ├── QUICKSTART.md           ← Start here! (10 min setup)
│   ├── PRODUCTION_GUIDE.md      ← Deep dive explanations
│   ├── ARCHITECTURE.md          ← This file
│   └── Dockerfile              ← Containerization
│
├── 🧠 CORE SOURCE CODE (src/)
│   ├── train.py                ← Main entry point for training
│   ├── api.py                  ← FastAPI web server
│   ├── utils.py                ← Logging, config, validation
│   ├── data_pipeline.py        ← Load, validate, augment data
│   ├── model_training.py       ← Build, train, evaluate model
│   └── inference.py            ← Make predictions
│
├── 📊 DATA DIRECTORY (data/)
│   ├── raw/                    ← Original images
│   │   ├── healthy/
│   │   ├── powdery_mildew/
│   │   ├── rust/
│   │   ├── leaf_spot/
│   │   └── blight/
│   └── processed/              ← (Optional) Preprocessed images
│
├── 🤖 MODELS DIRECTORY (models/)
│   └── trained/
│       ├── plant_classifier_v1.h5          ← Saved model weights
│       └── plant_classifier_v1_metadata.json ← Model info
│
├── ✅ TESTING (tests/)
│   └── test_pipeline.py        ← Comprehensive test suite
│
└── 📝 LOGS (logs/)
    ├── app.log                 ← Application logs
    └── errors.jsonl            ← Structured error logs
```

---

## How Each Component Works Together

### Data Flow: Training

```
1. config.yaml
   │
   ├─→ tells train.py where data is
   └─→ tells train.py what hyperparameters to use
   
2. train.py (main orchestrator)
   │
   ├─→ calls ImageDataLoader
   │   │ loads images from disk
   │   └─ validates format, size
   │
   ├─→ calls DataSplitter
   │   │ splits into train/val/test
   │   └─ maintains class distribution
   │
   ├─→ calls ModelBuilder
   │   │ creates ResNet50-based architecture
   │   └─ transfers learning from ImageNet
   │
   ├─→ calls ModelTrainer
   │   │ trains on training data
   │   │ validates on validation data
   │   └─ saves best model automatically
   │
   └─→ calls ModelEvaluator
       │ evaluates on test data
       └─ generates performance report

3. models/trained/plant_classifier_v1.h5
   └─ Saved model ready for inference
```

### Data Flow: Inference (API)

```
User Request
    │
    ↓
1. FastAPI receives HTTP POST request
   └─ File upload with image
   
    ↓
2. api.py validates:
   ├─ File exists?
   ├─ File size OK?
   ├─ File format OK? (JPG, PNG)
   └─ If any check fails → return error
   
    ↓
3. ImageValidator confirms image is valid
   
    ↓
4. ModelLoader returns cached model
   └─ Model loaded once at startup, reused for all requests
   
    ↓
5. ImagePredictor:
   ├─ Loads image from disk
   ├─ Preprocesses (resize, normalize)
   ├─ Passes to model
   ├─ Gets predictions
   └─ Formats response
   
    ↓
6. FastAPI returns JSON response:
   {
     "class": "powdery_mildew",
     "confidence": 0.92,
     "all_predictions": {...}
   }
   
    ↓
User gets prediction
```

---

## Design Patterns Used

### 1. Separation of Concerns
Each module has ONE responsibility:
```
utils.py          → Logging, config, validation helpers
data_pipeline.py  → Data loading, preprocessing, splitting
model_training.py → Model building, training, evaluation
inference.py      → Making predictions
api.py            → HTTP server and request handling
```

### 2. Dependency Injection
Pass dependencies instead of hardcoding:
```python
# ✗ Bad: Hardcoded dependency
class ModelTrainer:
    def __init__(self):
        self.config = load_config()  # Hardcoded path

# ✓ Good: Injected dependency
class ModelTrainer:
    def __init__(self, config):
        self.config = config  # Passed in, testable
```

### 3. Configuration Management
All settings in one file:
```yaml
data:
  image_size: 224
  batch_size: 32
model:
  learning_rate: 1e-4
  epochs: 20
```

No magic numbers in code!

### 4. Defensive Programming
Always validate input:
```python
def predict(file):
    if not file:
        raise HTTPException("No file")
    if file.size > MAX_SIZE:
        raise HTTPException("File too large")
    if not is_valid_format(file):
        raise HTTPException("Invalid format")
    # Now safe to process
```

### 5. Caching
Load expensive resources once:
```python
# Bad: Load model for every request (~1 sec × 100 requests = 100 sec waste)
# Good: Load once at startup (1 sec total)
@app.on_event("startup")
async def startup():
    global model
    model = load_model("path.h5")  # Once at startup
```

---

## Key Technologies & Why

| Technology | Purpose | Why This Choice |
|-----------|---------|-----------------|
| **TensorFlow/Keras** | Deep learning | Industry standard, pre-trained models |
| **ResNet50** | Base architecture | Fast, accurate, good for images |
| **Transfer Learning** | Quick training | Works with small datasets |
| **FastAPI** | Web server | Fast, automatic docs, validation |
| **Pydantic** | Data validation | Type-safe, automatic documentation |
| **Docker** | Containerization | Reproducible, portable, cloud-ready |
| **pytest** | Testing | Standard, comprehensive |
| **OpenCV** | Image processing | Industry standard for computer vision |
| **scikit-learn** | Metrics | Standard for ML evaluation |

---

## How I Think About Building This

### Phase 1: Architecture Design
```
Before writing code:
✓ Draw data flow diagram
✓ Identify components
✓ Plan how to test each part
✓ Plan failure modes
✓ Plan monitoring
```

### Phase 2: Build Components Independently
```
Each component in isolation:
✓ data_pipeline works without model
✓ model_training works without API
✓ inference works without training
✓ api can be tested without real data
```

### Phase 3: Integration
```
Connect components:
✓ config ties everything together
✓ train.py orchestrates pipeline
✓ api uses inference module
✓ tests verify integration
```

### Phase 4: Production Hardening
```
Make it robust:
✓ Error handling everywhere
✓ Logging for debugging
✓ Validation on all inputs
✓ Health checks for monitoring
✓ Container for deployment
```

---

## Production Checklist

### Development
- [ ] Code written and commented
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Local training works
- [ ] Local API works

### Quality Assurance
- [ ] Performance acceptable (<500ms per prediction)
- [ ] Error handling covers edge cases
- [ ] Logging shows what's happening
- [ ] Configuration can be changed without code changes

### Deployment
- [ ] Docker builds successfully
- [ ] Docker image runs locally
- [ ] Health check endpoint works
- [ ] API documentation complete
- [ ] Model versioning in place

### Monitoring
- [ ] Logging configured
- [ ] Error tracking configured
- [ ] Performance metrics tracked
- [ ] Alerting configured
- [ ] Rollback procedure documented

### Maintenance
- [ ] Retraining procedure documented
- [ ] Model versioning system in place
- [ ] Data versioning system in place
- [ ] Performance monitoring active
- [ ] Team trained on operation

---

## Common Patterns Explained

### Pattern 1: Config-Driven Development
```python
# GOOD: All settings in config.yaml
config = load_config()
learning_rate = config["model"]["learning_rate"]
epochs = config["model"]["epochs"]

# Can change settings without touching code!
```

### Pattern 2: Logging Not Printing
```python
# BAD: Hard to track in production
print("Training accuracy: 0.85")

# GOOD: Searchable, timestamped, organized
logger.info("Training accuracy: 0.85")
# Log file: 2024-01-15 10:30:45 - INFO - Training accuracy: 0.85
```

### Pattern 3: Validation Before Processing
```python
# BAD: Trust input
def predict(file):
    return model.predict(file)

# GOOD: Validate everything
def predict(file):
    if not file:
        raise HTTPException("No file")
    if not is_valid(file):
        raise HTTPException("Invalid")
    return model.predict(file)
```

### Pattern 4: Dependency Injection
```python
# BAD: Hard to test (depends on file system)
class Predictor:
    def __init__(self):
        self.model = load_model("path.h5")

# GOOD: Easy to test (inject mock model)
class Predictor:
    def __init__(self, model):
        self.model = model
```

---

## How to Extend This

### Add New Disease Class
1. Add images to `data/raw/new_disease/`
2. Update `classes` in `config.yaml`
3. Update `num_classes` in `config.yaml`
4. Retrain: `python src/train.py`

### Switch to Different Model
1. Change `model_name` in `config.yaml`
2. Update `ModelBuilder.build_model()` in `model_training.py`
3. Retrain: `python src/train.py`

### Add Authentication
1. Install: `pip install python-jose`
2. Add JWT validation in `api.py`
3. Require token in `/predict` endpoint

### Add Model Monitoring
1. Track predictions in database
2. Periodically evaluate model on new data
3. Alert if performance degrades
4. Trigger retraining if needed

---

## Performance Optimization Tips

### Training Faster
```yaml
# In config.yaml
data:
  image_size: 128  # Smaller = faster (trade accuracy)
model:
  batch_size: 64   # Larger = faster (needs more GPU RAM)
```

### Inference Faster
```python
# Use TensorFlow optimizations
model = tf.lite.TFLiteConverter.from_keras_model(model)
# Runs 10x faster on edge devices
```

### Memory Efficient
```python
# Use data generators (don't load all data to memory)
train_generator = DataGenerator(
    images, labels,
    batch_size=32  # Process 32 at a time
)
```

---

## Monitoring in Production

### Key Metrics to Track
```
1. Prediction latency (p50, p95, p99)
2. Error rate (failed predictions)
3. Model accuracy (on recent data)
4. System resources (GPU%, memory)
5. Request throughput
```

### How to Monitor
```python
# Option 1: Application metrics
logger.info(f"Prediction: {class} ({confidence:.2%}) in {elapsed_ms}ms")

# Option 2: Prometheus metrics
from prometheus_client import Counter, Histogram
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency', 'Prediction time')

# Option 3: External services
# Datadog, New Relic, Sentry, etc.
```

---

## Summary

This system shows how to build production ML applications:

1. **Configuration First**: One source of truth (config.yaml)
2. **Modular Design**: Each component independent and testable
3. **Defensive**: Validate everything, handle errors gracefully
4. **Observable**: Log, measure, monitor everything
5. **Reproducible**: Same config + data = same results
6. **Documented**: Code comments, architecture docs, guides
7. **Containerized**: Docker for easy deployment
8. **Tested**: Unit tests, integration tests, manual testing

This isn't overcomplicated. It's the **minimum** needed for production systems.

---

## Learning Resources

- **Read QUICKSTART.md**: Get running in 10 minutes
- **Read PRODUCTION_GUIDE.md**: Understand each component deeply
- **Read code comments**: I explain reasoning throughout
- **Modify and retrain**: Learn by experimentation
- **Deploy and monitor**: Learn what breaks in production

**You now have a template for any image classification project!** 🚀
