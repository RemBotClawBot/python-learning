#!/usr/bin/env python3
"""
Machine Learning Examples
=========================

This module demonstrates fundamental machine learning concepts using:
- Scikit-learn for traditional ML
- Basic neural networks with TensorFlow/Keras
- Model evaluation and validation
- Data preprocessing
- Common algorithms and use cases

Covers: regression, classification, clustering, dimensionality reduction,
neural networks, and model deployment basics.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from pathlib import Path

# Try to import ML libraries
try:
    from sklearn import datasets, model_selection, preprocessing, metrics
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Note: scikit-learn not installed. Run: pip install scikit-learn")

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Note: tensorflow not installed. Run: pip install tensorflow")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not installed. Run: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Note: seaborn not installed. Run: pip install seaborn")


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    if not SKLEARN_AVAILABLE:
        missing.append("scikit-learn")
    if not MATPLOTLIB_AVAILABLE:
        missing.append("matplotlib")
    if not SEABORN_AVAILABLE:
        missing.append("seaborn")
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    return True


def demo_linear_regression():
    """
    Demonstrate linear regression with scikit-learn.
    Predict house prices based on features.
    """
    print("\n" + "="*60)
    print("LINEAR REGRESSION DEMO")
    print("="*60)
    
    # Create synthetic housing data
    np.random.seed(42)
    n_samples = 100
    
    # Features: size (sq ft), bedrooms, age (years)
    X = np.column_stack([
        np.random.randint(800, 4000, n_samples),  # size
        np.random.randint(1, 6, n_samples),      # bedrooms
        np.random.randint(0, 50, n_samples)       # age
    ])
    
    # Target: price (in thousands)
    # Price = 100 * size + 50 * bedrooms - 5 * age + noise
    y = 100 * X[:, 0]/1000 + 50 * X[:, 1] - 5 * X[:, 2] + np.random.normal(0, 50, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = metrics.mean_squared_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    
    print(f"Model Coefficients:")
    print(f"  Size (per sq ft): ${model.coef_[0]:.2f}")
    print(f"  Bedrooms: ${model.coef_[1]:.2f}")
    print(f"  Age: ${model.coef_[2]:.2f}")
    print(f"  Intercept: ${model.intercept_:.2f}")
    print(f"\nPerformance Metrics:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R² Score: {r2:.2f}")
    
    # Plot predictions vs actual
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price ($k)')
        plt.ylabel('Predicted Price ($k)')
        plt.title('Actual vs Predicted Prices')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price ($k)')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ml_regression_demo.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved as 'ml_regression_demo.png'")
    
    return model, X_test, y_test, y_pred


def demo_classification():
    """
    Demonstrate classification with multiple algorithms.
    Use the classic Iris dataset.
    """
    print("\n" + "="*60)
    print("CLASSIFICATION DEMO (Iris Dataset)")
    print("="*60)
    
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Classes: {', '.join(target_names)}")
    
    # Split data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=3),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        # Train
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled) if hasattr(clf, 'predict_proba') else None
        
        # Evaluate
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': clf
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  CV Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    # Plot comparison
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(12, 5))
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, 2, i+1)
            values = [results[name][metric] for name in classifiers.keys()]
            bars = plt.bar(range(len(values)), values)
            plt.xticks(range(len(values)), list(classifiers.keys()), rotation=45, ha='right')
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} Comparison')
            plt.ylim([0.8, 1.0])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('ml_classification_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nComparison plot saved as 'ml_classification_comparison.png'")
    
    # Confusion matrix for best model
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_model_name]['model']
    y_pred_best = best_model.predict(X_test_scaled)
    
    if MATPLOTLIB_AVAILABLE and SEABORN_AVAILABLE:
        plt.figure(figsize=(8, 6))
        cm = metrics.confusion_matrix(y_test, y_pred_best)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('ml_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix for {best_model_name} saved as 'ml_confusion_matrix.png'")
    
    return results


def demo_clustering():
    """
    Demonstrate clustering algorithms (unsupervised learning).
    Use synthetic data for clear visualization.
    """
    print("\n" + "="*60)
    print("CLUSTERING DEMO")
    print("="*60)
    
    # Create synthetic data with 3 clusters
    np.random.seed(42)
    n_samples = 300
    
    # Generate 3 blobs
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=n_samples, centers=3, 
                          cluster_std=1.0, random_state=42)
    
    print(f"Generated {n_samples} samples with 3 true clusters")
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    # Apply DBSCAN (density-based)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    y_dbscan = dbscan.fit_predict(X)
    
    # Apply Agglomerative clustering
    from sklearn.cluster import AgglomerativeClustering
    agglo = AgglomerativeClustering(n_clusters=3)
    y_agglo = agglo.fit_predict(X)
    
    # Evaluate clustering
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    
    clustering_results = {}
    for name, labels in [('K-Means', y_kmeans), ('DBSCAN', y_dbscan), ('Agglomerative', y_agglo)]:
        if len(set(labels)) > 1:  # Silhouette needs at least 2 clusters
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = np.nan
        
        # Adjusted Rand Index compares to true labels (if known)
        ari = adjusted_rand_score(y_true, labels)
        
        clustering_results[name] = {
            'silhouette': silhouette,
            'ari': ari,
            'n_clusters': len(set(labels)),
            'labels': labels
        }
        
        print(f"\n{name}:")
        print(f"  Number of clusters found: {len(set(labels))}")
        print(f"  Silhouette Score: {silhouette:.3f}" if not np.isnan(silhouette) else "  Silhouette Score: N/A (only 1 cluster)")
        print(f"  Adjusted Rand Index: {ari:.3f}")
    
    # Visualize clustering results
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(15, 4))
        
        plots = [
            ('True Labels', y_true),
            ('K-Means', y_kmeans),
            ('DBSCAN', y_dbscan),
            ('Agglomerative', y_agglo)
        ]
        
        for i, (title, labels) in enumerate(plots):
            plt.subplot(1, 4, i+1)
            scatter = plt.scatter(X[:, 0], X[:, 1], c=labels,
                                cmap='viridis', s=50, alpha=0.8)
            plt.title(title)
            plt.xlabel('Feature 1')
            if i == 0:
                plt.ylabel('Feature 2')
            plt.colorbar(scatter, fraction=0.046, pad=0.04)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ml_clustering_demo.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nClustering visualization saved as 'ml_clustering_demo.png'")
    
    return clustering_results, X


def demo_pca():
    """
    Demonstrate Principal Component Analysis for dimensionality reduction.
    """
    print("\n" + "="*60)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA) DEMO")
    print("="*60)
    
    # Load wine dataset
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nPCA Results:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained by 2 components: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Visualize PCA results
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(12, 5))
        
        # PCA plot
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Wine Dataset PCA (2 components)')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        
        # Variance explained plot
        plt.subplot(1, 2, 2)
        pca_full = PCA().fit(X_scaled)
        variance_ratio = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        
        plt.plot(range(1, len(variance_ratio)+1), variance_ratio, 'bo-', label='Individual')
        plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'ro-', label='Cumulative')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Scree Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations for important points
        for i, (v, c) in enumerate(zip(variance_ratio[:5], cumulative_variance[:5]), 1):
            plt.annotate(f'{v:.2f}', (i, v), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(f'{c:.2f}', (i, c), textcoords="offset points", xytext=(0,-15), ha='center')
        
        plt.tight_layout()
        plt.savefig('ml_pca_demo.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPCA visualization saved as 'ml_pca_demo.png'")
    
    return pca, X_pca


def demo_neural_network():
    """
    Demonstrate a basic neural network using TensorFlow/Keras.
    Classification on the MNIST dataset (digits).
    """
    print("\n" + "="*60)
    print("NEURAL NETWORK DEMO (MNIST Digits)")
    print("="*60)
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Skipping neural network demo.")
        print("Install with: pip install tensorflow")
        return None
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess
    X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Build a simple neural network
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train_cat,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred_proba = model.predict(X_test[:10], verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print("\nPredictions for first 10 test images:")
    for i in range(10):
        print(f"  Image {i}: True={y_test[i]}, Predicted={y_pred[i]}, Confidence={y_pred_proba[i][y_pred[i]]:.2f}")
    
    # Plot training history
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ml_neural_network_training.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nTraining history plot saved as 'ml_neural_network_training.png'")
    
    return model, history, test_accuracy


def demo_model_pipeline():
    """
    Demonstrate a complete ML pipeline with preprocessing and grid search.
    """
    print("\n" + "="*60)
    print("ML PIPELINE DEMO WITH GRID SEARCH")
    print("="*60)
    
    # Load breast cancer dataset
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    print(f"Dataset: {data.DESCR[:200]}...")
    print(f"Features: {len(feature_names)}")
    print(f"Samples: {X.shape[0]}")
    print(f"Target: malignant (0) vs benign (1)")
    
    # Split data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameter grid for grid search
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search with cross-validation
    print("\nPerforming grid search with 5-fold cross-validation...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {test_accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(metrics.classification_report(y_test, y_pred, target_names=['malignant', 'benign']))
    
    # Feature importance from best model
    best_model = grid_search.best_estimator_.named_steps['classifier']
    feature_importance = best_model.feature_importances_
    
    # Get top 10 features
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = [feature_importance[i] for i in top_indices]
    
    print("\nTop 10 Most Important Features:")
    for feature, importance in zip(top_features, top_importance):
        print(f"  {feature}: {importance:.4f}")
    
    # Plot feature importance
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_importance[::-1], align='center')
        plt.yticks(y_pos, top_features[::-1])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('ml_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nFeature importance plot saved as 'ml_feature_importance.png'")
    
    return grid_search, test_accuracy


def create_model_serialization_example():
    """
    Demonstrate model saving and loading for deployment.
    """
    print("\n" + "="*60)
    print("MODEL SERIALIZATION DEMO")
    print("="*60)
    
    # Create a simple model
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    import joblib
    import pickle
    
    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Save model using joblib (recommended for scikit-learn)
    joblib.dump(model, 'logistic_regression_model.joblib')
    
    # Save model using pickle (alternative)
    with open('logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save model metadata
    import json
    model_metadata = {
        'model_type': 'LogisticRegression',
        'n_features': X.shape[1],
        'training_date': str(pd.Timestamp.now()),
        'accuracy': model.score(X, y)
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Load model
    model_loaded = joblib.load('logistic_regression_model.joblib')
    
    # Test loaded model
    X_test, y_test = make_classification(n_samples=10, n_features=5, random_state=123)
    predictions = model_loaded.predict(X_test)
    accuracy = model_loaded.score(X_test, y_test)
    
    print(f"Model trained on {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Model saved to: logistic_regression_model.joblib")
    print(f"Model metadata saved to: model_metadata.json")
    print(f"Loaded model test accuracy: {accuracy:.4f}")
    print(f"Predictions on new data: {predictions}")
    
    # Clean up
    for file in ['logistic_regression_model.joblib', 
                 'logistic_regression_model.pkl',
                 'model_metadata.json']:
        if os.path.exists(file):
            os.remove(file)
    
    return model_loaded


def main():
    """Run all machine learning demonstrations."""
    print("="*60)
    print("MACHINE LEARNING EXAMPLES")
    print("="*60)
    print("\nThis module demonstrates fundamental ML concepts:")
    print("1. Linear Regression")
    print("2. Classification (Multiple Algorithms)")
    print("3. Clustering (Unsupervised Learning)")
    print("4. Principal Component Analysis (Dimensionality Reduction)")
    print("5. Neural Networks (TensorFlow/Keras)")
    print("6. Complete ML Pipeline with Grid Search")
    print("7. Model Serialization for Deployment")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies to run all demos.")
        return
    
    # Run demos
    results = {}
    
    try:
        print("\n" + "="*60)
        print("1. LINEAR REGRESSION")
        print("="*60)
        results['regression'] = demo_linear_regression()
    except Exception as e:
        print(f"Regression demo failed: {e}")
    
    try:
        print("\n" + "="*60)
        print("2. CLASSIFICATION")
        print("="*60)
        results['classification'] = demo_classification()
    except Exception as e:
        print(f"Classification demo failed: {e}")
    
    try:
        print("\n" + "="*60)
        print("3. CLUSTERING")
        print("="*60)
        results['clustering'] = demo_clustering()
    except Exception as e:
        print(f"Clustering demo failed: {e}")
    
    try:
        print("\n" + "="*60)
        print("4. PRINCIPAL COMPONENT ANALYSIS")
        print("="*60)
        results['pca'] = demo_pca()
    except Exception as e:
        print(f"PCA demo failed: {e}")
    
    try:
        print("\n" + "="*60)
        print("5. NEURAL NETWORKS")
        print("="*60)
        results['neural_network'] = demo_neural_network()
    except Exception as e:
        print(f"Neural network demo failed: {e}")
    
    try:
        print("\n" + "="*60)
        print("6. ML PIPELINE WITH GRID SEARCH")
        print("="*60)
        results['pipeline'] = demo_model_pipeline()
    except Exception as e:
        print(f"Pipeline demo failed: {e}")
    
    try:
        print("\n" + "="*60)
        print("7. MODEL SERIALIZATION")
        print("="*60)
        results['serialization'] = create_model_serialization_example()
    except Exception as e:
        print(f"Serialization demo failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("All ML demonstrations completed!")
    print("\nGenerated files:")
    
    # List generated files
    ml_files = [f for f in os.listdir('.') if f.startswith('ml_') and f.endswith('.png')]
    for file in ml_files:
        print(f"  • {file}")
    
    print("\nTo explore further:")
    print("1. Install all dependencies: pip install scikit-learn tensorflow matplotlib seaborn")
    print("2. Run individual functions: python -c 'import machine_learning_examples; machine_learning_examples.demo_classification()'")
    print("3. Modify parameters and experiment with different algorithms")
    print("4. Try with your own datasets!")
    
    return results


if __name__ == "__main__":
    main()