#!/usr/bin/env python3
"""
Intent Prediction Model Training
Smart Traffic Management System - Phase 2

This script preprocesses vehicle data collected from SUMO simulation
and trains a machine learning model to predict vehicle turning intent.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse


class IntentPredictor:
    """Vehicle turning intent prediction model"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.target_column = 'intent'
        
    def preprocess_data(self, df):
        """Preprocess the vehicle data for training"""
        print("Preprocessing data...")
        
        # Remove records with unknown intent
        df = df[df['intent'] != 'unknown'].copy()
        
        # Create additional features
        df['speed_acceleration_ratio'] = df['speed'] / (df['acceleration'].abs() + 0.001)
        df['distance_speed_ratio'] = df['distance_to_intersection'] / (df['speed'] + 0.001)
        
        # Encode lane information
        df['lane_direction'] = df['lane_id'].str.extract(r'([a-z]+)_')[0]
        
        # Create lane direction encoding
        lane_direction_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        df['lane_direction_encoded'] = df['lane_direction'].map(lane_direction_map).fillna(-1)
        
        # Create vehicle type encoding
        vehicle_type_map = {'car': 0, 'truck': 1, 'bus': 2, 'emergency': 3}
        df['vehicle_type_encoded'] = df['vehicle_type'].map(vehicle_type_map).fillna(0)
        
        # Calculate speed derivatives (approximation)
        df_sorted = df.sort_values(['vehicle_id', 'step'])
        df_sorted['speed_change'] = df_sorted.groupby('vehicle_id')['speed'].diff().fillna(0)
        df_sorted['acceleration_change'] = df_sorted.groupby('vehicle_id')['acceleration'].diff().fillna(0)
        
        # Features that will be used for training
        feature_columns = [
            'x', 'y', 'speed', 'acceleration', 'lane_position',
            'distance_to_intersection', 'waiting_time', 'angle',
            'speed_acceleration_ratio', 'distance_speed_ratio',
            'lane_direction_encoded', 'vehicle_type_encoded',
            'speed_change', 'acceleration_change'
        ]
        
        # Remove rows with NaN values
        df_clean = df_sorted[feature_columns + [self.target_column]].dropna()
        
        print(f"Data shape after preprocessing: {df_clean.shape}")
        print(f"Intent distribution after preprocessing:")
        print(df_clean[self.target_column].value_counts())
        
        return df_clean, feature_columns
    
    def prepare_features_and_target(self, df, feature_columns):
        """Prepare features and target for training"""
        X = df[feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.feature_columns = feature_columns
        
        return X_scaled, y_encoded, y
    
    def train_model(self, X, y, model_type='random_forest'):
        """Train the intent prediction model"""
        print(f"Training {model_type} model...")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        self.model.fit(X, y)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, y_test_original):
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Convert back to original labels for reporting
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_labels))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred_labels,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test_labels, y_pred_labels, output_dict=True)
        }
    
    def plot_results(self, results, output_dir):
        """Plot evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Intent Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title('Feature Importance - Intent Prediction')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        print(f"Performing {cv}-fold cross-validation...")
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def save_model(self, output_dir, model_name="intent_predictor"):
        """Save the trained model and preprocessors"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_file = os.path.join(output_dir, f"{model_name}.pkl")
        scaler_file = os.path.join(output_dir, f"{model_name}_scaler.pkl")
        encoder_file = os.path.join(output_dir, f"{model_name}_encoder.pkl")
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        joblib.dump(self.label_encoder, encoder_file)
        
        # Save feature columns
        feature_file = os.path.join(output_dir, f"{model_name}_features.txt")
        with open(feature_file, 'w') as f:
            f.write('\n'.join(self.feature_columns))
        
        print(f"Model saved to: {model_file}")
        print(f"Scaler saved to: {scaler_file}")
        print(f"Encoder saved to: {encoder_file}")
        print(f"Features saved to: {feature_file}")
    
    def load_model(self, model_dir, model_name="intent_predictor"):
        """Load a trained model and preprocessors"""
        model_file = os.path.join(model_dir, f"{model_name}.pkl")
        scaler_file = os.path.join(model_dir, f"{model_name}_scaler.pkl")
        encoder_file = os.path.join(model_dir, f"{model_name}_encoder.pkl")
        feature_file = os.path.join(model_dir, f"{model_name}_features.txt")
        
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.label_encoder = joblib.load(encoder_file)
        
        with open(feature_file, 'r') as f:
            self.feature_columns = [line.strip() for line in f]
        
        print(f"Model loaded from: {model_file}")
    
    def predict_intent(self, vehicle_data):
        """Predict intent for new vehicle data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Ensure data has all required features
        for col in self.feature_columns:
            if col not in vehicle_data.columns:
                raise ValueError(f"Missing feature: {col}")
        
        # Scale features
        X = vehicle_data[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled) if hasattr(self.model, 'predict_proba') else None
        
        # Convert back to original labels
        predictions = self.label_encoder.inverse_transform(y_pred)
        
        return predictions, y_pred_proba


def load_and_combine_data(data_dir):
    """Load and combine data from multiple CSV files"""
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    print(f"Found {len(csv_files)} data files:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Load and combine all CSV files
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        df['source_file'] = file
        dataframes.append(df)
        print(f"Loaded {file}: {len(df)} records")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} records")
    
    return combined_df


def main():
    """Main function to train intent prediction model"""
    parser = argparse.ArgumentParser(description='Train vehicle intent prediction model')
    parser.add_argument('--data-dir', default='../data', help='Directory containing vehicle data CSV files')
    parser.add_argument('--output-dir', default='../ml', help='Output directory for models and results')
    parser.add_argument('--model-type', default='random_forest', 
                       choices=['random_forest', 'gradient_boosting', 'svm'],
                       help='Type of model to train')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SMART TRAFFIC MANAGEMENT - INTENT PREDICTION TRAINING")
    print("=" * 60)
    
    # Load data
    try:
        df = load_and_combine_data(args.data_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize predictor
    predictor = IntentPredictor()
    
    # Preprocess data
    df_clean, feature_columns = predictor.preprocess_data(df)
    
    # Prepare features and target
    X, y, y_original = predictor.prepare_features_and_target(df_clean, feature_columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    _, _, y_train_orig, y_test_orig = train_test_split(
        X, y_original, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    predictor.train_model(X_train, y_train, args.model_type)
    
    # Cross-validation
    cv_scores = predictor.cross_validate(X_train, y_train, args.cv_folds)
    
    # Evaluate model
    results = predictor.evaluate_model(X_test, y_test, y_test_orig)
    
    # Generate plots if requested
    if args.plot:
        predictor.plot_results(results, args.output_dir)
    
    # Save model
    model_name = f"intent_predictor_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    predictor.save_model(args.output_dir, model_name)
    
    # Save training summary
    summary_file = os.path.join(args.output_dir, f"{model_name}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Intent Prediction Model Training Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Dataset Size: {len(df_clean)} records\n")
        f.write(f"Features: {len(feature_columns)}\n")
        f.write(f"Test Size: {args.test_size}\n")
        f.write(f"CV Folds: {args.cv_folds}\n\n")
        f.write(f"Results:\n")
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
        f.write(f"Feature Columns:\n")
        for i, col in enumerate(feature_columns, 1):
            f.write(f"{i:2d}. {col}\n")
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved as: {model_name}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
