import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import time
import json
from pathlib import Path

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# File paths
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'new_movie.csv')
model_dir = os.path.join(current_dir, 'models')
os.makedirs(model_dir, exist_ok=True)

def load_and_preprocess_data(csv_path):
    """Load and preprocess movie data for recommendation"""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Data exploration
    print(f"Dataset shape: {df.shape}")
    print(f"Data distribution: \n{df['rating'].value_counts(normalize=True).sort_index()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Missing values detected: \n{missing_values[missing_values > 0]}")
        
        # Fill missing values appropriately
        # For numerical columns
        df['year'].fillna(df['year'].median(), inplace=True)
        df['month'].fillna(df['month'].median(), inplace=True)
        df['day'].fillna(df['day'].median(), inplace=True)
        
        # For categorical columns
        df['age'] = df['age'].fillna(df['age'].mode()[0]).astype(str)
        df['gender'] = df['gender'].fillna('Unknown').astype(str)
        df['occupation'] = df['occupation'].fillna('Other').astype(str)
        
        print(f"After handling missing values: {df.shape}")
    
    # Split features and target
    X = df.drop(['rating', 'title', 'movie_id'], axis=1)
    y = df['rating'].values
    
    # Keep only the columns we need
    X = X[['year', 'month', 'day', 'age', 'gender', 'occupation']]
    
    # Ensure categorical columns are string type
    X['age'] = X['age'].astype(str)
    X['gender'] = X['gender'].astype(str)
    X['occupation'] = X['occupation'].astype(str)
    
    # Identify column types
    categorical_columns = ['age', 'gender', 'occupation']
    numerical_columns = ['year', 'month', 'day']
    
    # Advanced preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', PowerTransformer(method='yeo-johnson', standardize=True), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
        ])
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    print(f"Input dimension after preprocessing: {X_processed.shape[1]}")
    
    return df, X_processed, y, preprocessor

def create_data_loaders(X_processed, y, batch_size=64, use_sampler=True):
    """Create train, validation and test data loaders with optional weighted sampling"""
    # Create stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=RANDOM_SEED, 
        stratify=np.round(y).astype(int))
    
    # Further split training data to create validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_SEED,
        stratify=np.round(y_train).astype(int))
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    # Weighted sampler for imbalanced rating distribution
    if use_sampler:
        # Create class weights based on rating frequencies
        y_train_rounded = np.round(y_train).astype(int)
        class_counts = np.bincount(y_train_rounded)
        class_weights = 1. / class_counts
        weights = class_weights[y_train_rounded]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False  # Don't shuffle when using sampler
    else:
        sampler = None
        shuffle = True
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=shuffle, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, X_train.shape[1]

class EnhancedRecommenderNN(nn.Module):
    """Enhanced Neural Network architecture for movie recommendations"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(EnhancedRecommenderNN, self).__init__()
        
        self.feature_extraction = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),  # Using GELU activation instead of LeakyReLU
            nn.Dropout(dropout_rate)
        )
        
        # Middle layers
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout_rate * (1 - 0.1*i))  # Decreasing dropout rate
            ])
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Skip connection
        self.skip_connection = nn.Linear(input_dim, hidden_dims[-1])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        identity = x
        
        # Feature extraction
        x = self.feature_extraction(x)
        
        # Hidden layers
        x = self.hidden_layers(x)
        
        # Skip connection
        x = x + self.skip_connection(identity)
        
        # Output
        x = self.output_layer(x)
        
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler=None, epochs=50, patience=10, device='cpu', 
                model_path=None, verbose=True):
    """Train model with early stopping and best model saving"""
    
    model = model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0
    
    # Create and log a time-stamped training run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(model_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up training metrics logging
    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Update learning rate if scheduler is provided
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Update metrics
        metrics['epochs'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['learning_rate'].append(current_lr)
        
        # Print progress
        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save best model
            best_model = model.state_dict().copy()
            
            # Save the best model
            if model_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, model_path)
                
                if verbose:
                    print(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Calculate training time
    total_time = time.time() - start_time
    
    # Log metrics
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    # Create and save loss plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epochs'], metrics['learning_rate'])
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_plot.png'))
    
    if verbose:
        print(f"Training completed in {total_time:.2f} seconds")
    
    # Load best model
    model.load_state_dict(best_model)
    return model, train_losses, val_losses, best_val_loss, metrics

def evaluate_model(model, data_loader, criterion, device='cpu'):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    actuals = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy().flatten())
            
    avg_loss = total_loss / len(data_loader.dataset)
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    
    # Create error distribution plot
    errors = np.array(predictions) - np.array(actuals)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(actuals, predictions, alpha=0.3)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Actual vs Predicted Ratings')
    
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title(f'Error Distribution (μ={np.mean(errors):.4f}, σ={np.std(errors):.4f})')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png')
    
    return {
        'loss': avg_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'actuals': actuals,
        'errors': errors
    }

def predict_rating(user_features, model, preprocessor, device='cpu'):
    """Make prediction for a new user"""
    # Preprocess input features
    if isinstance(user_features, dict):
        user_features = pd.DataFrame([user_features])
    
    # Ensure categorical columns are string type
    user_features['age'] = user_features['age'].astype(str)
    user_features['gender'] = user_features['gender'].astype(str)
    user_features['occupation'] = user_features['occupation'].astype(str)
    
    # Transform features
    user_features = preprocessor.transform(user_features)
    user_tensor = torch.FloatTensor(user_features).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        prediction = model(user_tensor)
    
    # Clamp predictions between 1 and 5
    prediction = torch.clamp(prediction, 1.0, 5.0)
    
    return prediction.cpu().numpy().flatten()

def main():
    # Load and preprocess data
    df, X_processed, y, preprocessor = load_and_preprocess_data(csv_path)
    
    # Create data loaders
    train_loader, val_loader, test_loader, input_dim = create_data_loaders(
        X_processed, y, batch_size=128, use_sampler=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Input dimension: {input_dim}")
    
    # Experiment configurations
    experiments = {
        'baseline': {
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.3,
            'lr': 3e-3,
            'weight_decay': 1e-4,
            'batch_size': 128
        },
        'deeper': {
            'hidden_dims': [512, 256, 128, 64],
            'dropout_rate': 0.3,
            'lr': 2e-3,
            'weight_decay': 1e-4,
            'batch_size': 128
        }
    }
    
    # Choose experiment
    exp_name = 'deeper'  # or 'baseline'
    config = experiments[exp_name]
    
    # Initialize model with config
    model = EnhancedRecommenderNN(
        input_dim=input_dim,
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.SmoothL1Loss()  # Huber loss for robustness
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler - One Cycle LR
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        epochs=50,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # Warm up period
        anneal_strategy='cos',
        final_div_factor=10
    )
    
    # Model path - Changed to movie_finder.pth
    model_path = os.path.join(model_dir, 'movie_finder.pth')
    
    # Train the model
    model, train_losses, val_losses, best_val_loss, metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=50,
        patience=10,
        device=device,
        model_path=model_path
    )
    
    # Evaluate on test set
    test_results = evaluate_model(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"MSE: {test_results['mse']:.4f}")
    print(f"RMSE: {test_results['rmse']:.4f}")
    print(f"MAE: {test_results['mae']:.4f}")
    print(f"R² Score: {test_results['r2']:.4f}")
    
    # Example usage with different types of users
    example_users = [
        {
            'year': 2023, 'month': 5, 'day': 15,
            'age': 30, 'gender': 'M', 'occupation': 'Engineer'
        },
        {
            'year': 2023, 'month': 6, 'day': 20,
            'age': 25, 'gender': 'F', 'occupation': 'Student'
        },
        {
            'year': 2023, 'month': 7, 'day': 10,
            'age': 45, 'gender': 'F', 'occupation': 'Executive'
        }
    ]
    
    for i, user in enumerate(example_users):
        # Convert to DataFrame with the right column structure
        user_df = pd.DataFrame([user])
        
        # Make prediction
        predicted_rating = predict_rating(user_df, model, preprocessor, device)
        print(f"User {i+1} Predicted Rating: {predicted_rating[0]:.2f}")
    
    # Save preprocessor
    import joblib
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
    
    print("Model and preprocessor saved successfully!")
    
    # Return model and preprocessor for use in API
    return model, preprocessor

if __name__ == "__main__":
    main()