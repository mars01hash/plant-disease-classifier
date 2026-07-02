"""
Model Training for Image Classification.

Production approach: Transfer Learning
- Use pre-trained model (ResNet50 trained on ImageNet)
- Replace final layer for our classes
- Train only the new layer (fine-tuning)

Why this works:
- Pre-trained model already knows how to detect edges, shapes, textures
- We only need to learn what's specific to our disease classification
- Requires much less data and training time
- Generally performs better than training from scratch
"""

import os
import numpy as np
from typing import Dict, Tuple, List
import logging
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

# ============================================================================
# MODEL BUILDER
# ============================================================================

class ModelBuilder:
    """
    Build transfer learning model.
    
    Architecture:
    Input → Pre-trained ResNet50 → Global Average Pooling → 
    Dropout → Dense(num_classes) → Softmax
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.image_size = config["data"]["image_size"]
        self.num_classes = config["model"]["num_classes"]
    
    def build_model(self) -> keras.Model:
        """
        Build transfer learning model.
        
        Production insight: Transfer learning is the standard approach for
        computer vision with limited data. Here's why:
        1. Pre-trained weights capture general visual features
        2. Only train new layer (much faster)
        3. Need fewer images to achieve good performance
        4. Better generalization to unseen data
        """
        self.logger.info("Building transfer learning model")
        
        # Input layer
        inputs = keras.Input(shape=(self.image_size, self.image_size, 3))
        
        # Load pre-trained ResNet50 (ImageNet weights)
        # include_top=False removes the classification layer
        base_model = ResNet50(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model weights (don't train them)
        # We only want to train the new layers we add
        if self.config["model"]["freeze_backbone"]:
            base_model.trainable = False
            self.logger.info("Froze base model weights (will not be updated during training)")
        
        # Pass input through base model
        x = base_model(inputs, training=False)
        
        # Global average pooling: reduce (H, W, C) → (C,)
        # This gives us one number per feature channel
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dropout: randomly zero out 50% of activations during training
        # This prevents overfitting (model doesn't rely on any single feature)
        x = layers.Dropout(self.config["model"]["dropout_rate"])(x)
        
        # Classification layer: map to num_classes
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs, outputs)
        
        self.logger.info(f"Model built successfully")
        self.logger.info(f"Total parameters: {model.count_params():,}")
        self.logger.info(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
    
    def compile_model(self, model: keras.Model):
        """
        Compile model (set loss, optimizer, metrics).
        
        Production insight: These settings heavily influence training.
        - Learning rate: too high = unstable training, too low = slow convergence
        - Loss: categorical_crossentropy for multi-class classification
        - Metrics: what do we measure? accuracy is standard
        """
        learning_rate = self.config["model"]["learning_rate"]
        
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',  # For one-hot encoded labels
            metrics=[
                keras.metrics.CategoricalAccuracy(),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
            ]
        )
        
        self.logger.info(f"Model compiled with:")
        self.logger.info(f"  Optimizer: Adam (lr={learning_rate})")
        self.logger.info(f"  Loss: categorical_crossentropy")
        self.logger.info(f"  Metrics: accuracy, precision, recall")


# ============================================================================
# MODEL TRAINER
# ============================================================================

class ModelTrainer:
    """
    Train the model with proper callbacks and monitoring.
    
    Production insight: Always use callbacks:
    - Early stopping: stop when validation stops improving
    - Model checkpoint: save best model automatically
    - Learning rate reduction: decrease LR when stuck
    - TensorBoard: visualize training in real-time
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.history = None
    
    def get_callbacks(self) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks.
        
        Each callback serves a purpose:
        1. EarlyStopping: Stop when validation accuracy stops improving
        2. ModelCheckpoint: Save the best model
        3. ReduceLROnPlateau: Reduce learning rate if stuck
        4. TensorBoard: Monitor training in TensorBoard
        """
        model_path = self.config["model"]["save_path"]
        
        callbacks = [
            # Stop if validation accuracy doesn't improve for 5 epochs
            EarlyStopping(
                monitor='val_categorical_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Save model when validation accuracy improves
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_categorical_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate by 0.5x if validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
        ]
        
        return callbacks
    
    def train(self, model: keras.Model, 
              train_generator, val_generator,
              epochs: int) -> Dict:
        """
        Train the model.
        
        Production considerations:
        - Use generators to avoid loading all data to memory
        - Monitor both training and validation metrics
        - Stop early if overfitting occurs
        - Save model automatically
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Training samples: {len(train_generator)}")
        self.logger.info(f"Validation samples: {len(val_generator)}")
        
        callbacks = self.get_callbacks()
        
        # Train model
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        
        # Log final metrics
        final_metrics = {
            'train_loss': float(history.history['loss'][-1]),
            'train_accuracy': float(history.history['categorical_accuracy'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'val_accuracy': float(history.history['val_categorical_accuracy'][-1]),
            'epochs_trained': len(history.history['loss']),
        }
        
        self.logger.info("Training completed!")
        self.logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
        
        return final_metrics


# ============================================================================
# MODEL EVALUATOR
# ============================================================================

class ModelEvaluator:
    """
    Evaluate model on test set and generate performance report.
    
    Production insight: Always evaluate on test set AFTER training.
    Test set represents real-world performance.
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def evaluate(self, model: keras.Model, 
                 test_images: np.ndarray, test_labels_onehot: np.ndarray,
                 test_labels: np.ndarray, classes: List[str]) -> Dict:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary with:
            - Overall metrics (accuracy, precision, recall)
            - Per-class metrics (precision, recall, F1 for each disease)
            - Confusion matrix
        """
        self.logger.info("Evaluating model on test set")
        
        # Get predictions
        predictions = model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Overall metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, confusion_matrix, classification_report
        )
        
        accuracy = accuracy_score(test_labels, predicted_labels)
        precision = precision_score(test_labels, predicted_labels, average='weighted')
        recall = recall_score(test_labels, predicted_labels, average='weighted')
        f1 = f1_score(test_labels, predicted_labels, average='weighted')
        
        # Per-class metrics
        class_report = classification_report(
            test_labels, predicted_labels,
            target_names=classes,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(test_labels, predicted_labels)
        
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
            },
            'per_class': class_report,
            'confusion_matrix': conf_matrix.tolist(),
        }
        
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST SET EVALUATION")
        self.logger.info("="*60)
        self.logger.info(f"Accuracy:  {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall:    {recall:.4f}")
        self.logger.info(f"F1 Score:  {f1:.4f}")
        self.logger.info("\nPer-class metrics:")
        self.logger.info(classification_report(test_labels, predicted_labels, 
                                              target_names=classes))
        self.logger.info("="*60 + "\n")
        
        return metrics


# ============================================================================
# FULL TRAINING PIPELINE
# ============================================================================

def train_full_pipeline(config: Dict, logger: logging.Logger,
                       train_images: np.ndarray, train_labels: np.ndarray,
                       val_images: np.ndarray, val_labels: np.ndarray,
                       test_images: np.ndarray, test_labels: np.ndarray) -> Dict:
    """
    Execute full training pipeline.
    
    This is the main entry point for training.
    """
    from data_pipeline import DataGenerator, DataAugmentor
    
    # Build model
    builder = ModelBuilder(config, logger)
    model = builder.build_model()
    builder.compile_model(model)
    
    # Create data generators with augmentation for training
    augmentor = DataAugmentor(config, logger)
    train_generator = DataGenerator(
        train_images, train_labels,
        batch_size=config["data"]["batch_size"],
        augment=True,
        augmentor=augmentor
    )
    
    val_generator = DataGenerator(
        val_images, val_labels,
        batch_size=config["data"]["batch_size"],
        augment=False
    )
    
    # Train model
    trainer = ModelTrainer(config, logger)
    train_metrics = trainer.train(
        model, train_generator, val_generator,
        epochs=config["model"]["epochs"]
    )
    
    # Evaluate on test set
    # Convert test labels to one-hot
    test_labels_onehot = np.eye(config["model"]["num_classes"])[test_labels]
    
    evaluator = ModelEvaluator(config, logger)
    test_metrics = evaluator.evaluate(
        model, test_images, test_labels_onehot, test_labels, config["classes"]
    )
    
    # Combine metrics
    all_metrics = {**train_metrics, 'test': test_metrics}
    
    return {
        'model': model,
        'metrics': all_metrics,
        'history': trainer.history
    }
