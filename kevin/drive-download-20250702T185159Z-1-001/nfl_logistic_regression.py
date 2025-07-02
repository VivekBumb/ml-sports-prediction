"""
Logistic Regression Implementation as discussed in class 
========================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Check if required packages are available
print("=== CHECKING DEPENDENCIES ===")
try:
    print("pandas:", pd.__version__)
    print("numpy:", np.__version__)
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install pandas numpy")
    exit(1)

# Try to import matplotlib for plotting, skip if not available
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
    print("matplotlib: Available for plots")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("matplotlib: Not available - plots will be skipped")

print("All dependencies ready!\n")

def accuracy_score(y_true, y_pred):
    """Calculate accuracy score"""
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    """Create confusion matrix"""
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    
    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    
    return matrix

def classification_report(y_true, y_pred, target_names=None):
    """Generate classification report"""
    if target_names is None:
        target_names = ['Class 0', 'Class 1']
    
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    report = "\n"
    report += f"{'':>12} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}\n"
    report += "\n"
    
    for i, label in enumerate(unique_labels):
        true_positives = np.sum((y_true == label) & (y_pred == label))
        false_positives = np.sum((y_true != label) & (y_pred == label))
        false_negatives = np.sum((y_true == label) & (y_pred != label))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true == label)
        
        name = target_names[i] if i < len(target_names) else f'Class {label}'
        report += f"{name:>12} {precision:>9.2f} {recall:>9.2f} {f1:>9.2f} {support:>9}\n"
    
    return report

class LogisticRegressionFromScratch:
    """
    Logistic Regression implementation as per class lecture
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate  # η in slides
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None  # θ parameters
        self.cost_history = []
        
    def sigmoid(self, z):
        """
        Sigmoid function g(s) = 1/(1 + exp(-s)) from slide 19
        Handles overflow by clipping extreme values
        """
        # Clip z to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y):
        """
        Compute log-likelihood cost function from slides 23-24
        l(θ) = log∏p(y^{i}|x^{i}, θ)
        """
        m = X.shape[0]
        
        # Linear combination: s = X*θ
        linear_pred = X.dot(self.weights)
        
        # Predictions using sigmoid: g(s) 
        predictions = self.sigmoid(linear_pred)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Log-likelihood (we want to maximize this, so minimize negative)
        cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return cost
    
    def compute_gradient(self, X, y):
        """
        Compute gradient:
        ∂l(θ)/∂θ = Σᵢ (x^{i})^T (y^{i} - 1) + (x^{i})^T * exp(-x^{i}θ)/(1 + exp(-x^{i}θ))
        
        Simplified form: ∂l(θ)/∂θ = X^T(y - h_θ(x))
        """
        m = X.shape[0]
        
        # Linear combination: s = X*θ
        linear_pred = X.dot(self.weights)
        
        # Sigmoid predictions: h_θ(x) = g(X*θ)
        predictions = self.sigmoid(linear_pred)
        
        # Gradient calculation (for maximizing log-likelihood)
        gradient = X.T.dot(y - predictions) / m
        return gradient
    
    def fit(self, X, y):
        """
        Gradient Ascent Algorithm from slides 26-27
        θ^{t+1} ← θ^t + η * ∇l(θ)
        """
        # Add bias term (intercept) - slide shows θ₀ term
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Initialize weights (θ parameters)
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features)
        
        # Gradient Ascent iteration
        for i in range(self.max_iterations):
            # Calculate cost (negative log-likelihood)
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            # Calculate gradient
            gradient = self.compute_gradient(X, y)
            
            # Update weights: θ^{t+1} ← θ^t + η * ∇l(θ)
            # (Using + because we computed gradient for maximizing log-likelihood)
            new_weights = self.weights + self.learning_rate * gradient
            
            # Check convergence (slide 27 shows ||θ^{t+1} - θ^t|| > ε)
            if np.linalg.norm(new_weights - self.weights) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
                
            self.weights = new_weights
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}, Cost: {cost:.6f}")
    
    def predict_proba(self, X):
        """
        Predict probabilities: P(y=1|x) = g(X*θ)
        """
        # Add bias term
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Linear combination
        linear_pred = X.dot(self.weights)
        
        # Sigmoid transformation
        probabilities = self.sigmoid(linear_pred)
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions based on threshold (as per lecture 0.5 threshold)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def plot_cost_history(self):
        """
        Plot cost function convergence (as per lecture show the concave nature)
        """
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available - cost history saved to CSV instead")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Convergence (Log-Likelihood)')
        plt.xlabel('Iterations')
        plt.ylabel('Cost (Negative Log-Likelihood)')
        plt.grid(True)
        # Save plot
        plt.savefig('cost_convergence_plot.png', dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.1)

# Load and prepare NFL betting data
print("=== NFL BETTING PREDICTION WITH LOGISTIC REGRESSION ===")
print("Following Class Lecture\n")

# Load the preprocessed data
X = pd.read_csv('betting_features_X.csv')
y = pd.read_csv('betting_targets_y.csv')['home_win'].values

print(f"Dataset shape: {X.shape}")
print(f"Features: {X.shape[1]}")
print(f"Home win rate: {y.mean():.1%}\n")

# Split data chronologically (important for time series like sports data)
# Use last 20% as test set
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training set: {X_train.shape[0]} games")
print(f"Test set: {X_test.shape[0]} games")

# Convert to numpy arrays for our implementation
X_train_np = X_train.values
X_test_np = X_test.values

# Initialize and train logistic regression
print("\n=== TRAINING LOGISTIC REGRESSION ===")
lr_model = LogisticRegressionFromScratch(
    learning_rate=0.1,    # η from class lecture
    max_iterations=1000,
    tolerance=1e-6
)

# Fit the model using gradient ascent
lr_model.fit(X_train_np, y_train)

# Make predictions
print("\n=== MAKING PREDICTIONS ===")
train_predictions = lr_model.predict(X_train_np)
test_predictions = lr_model.predict(X_test_np)

# Get probabilities for analysis
train_probabilities = lr_model.predict_proba(X_train_np)
test_probabilities = lr_model.predict_proba(X_test_np)

# Evaluate performance
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Detailed evaluation
print("\n=== DETAILED EVALUATION ===")
classification_rep = classification_report(y_test, test_predictions, 
                          target_names=['Away Win', 'Home Win'])
print("Test Set Classification Report:")
print(classification_rep)

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, test_predictions)
print(cm)

# SAVE RESULTS TO FILES
print("\n=== SAVING RESULTS TO FILES ===")

# 1. Save model predictions with probabilities
results_df = pd.DataFrame({
    'actual_outcome': y_test,
    'predicted_outcome': test_predictions,
    'win_probability': test_probabilities,
    'correct_prediction': (y_test == test_predictions).astype(int),
    'confidence_level': ['HIGH' if p > 0.65 or p < 0.35 else 'MEDIUM' if p > 0.55 or p < 0.45 else 'LOW' 
                        for p in test_probabilities]
})
results_df.to_csv('logistic_regression_predictions.csv', index=False)
print("Saved predictions to: logistic_regression_predictions.csv")

# 2. Save feature importance
feature_importance_df = pd.DataFrame({
    'feature': ['bias_term'] + list(X.columns),
    'weight': lr_model.weights,
    'abs_weight': np.abs(lr_model.weights),
    'importance_rank': range(1, len(lr_model.weights) + 1)
}).sort_values('abs_weight', ascending=False)
feature_importance_df.to_csv('feature_importance.csv', index=False)
print("Saved feature weights to: feature_importance.csv")

# 3. Save performance metrics
performance_metrics = {
    'metric': ['train_accuracy', 'test_accuracy', 'total_games', 'correct_predictions', 
               'home_win_rate', 'model_bias', 'convergence_iterations'],
    'value': [train_accuracy, test_accuracy, len(y_test), sum(y_test == test_predictions),
              y.mean(), lr_model.weights[0], len(lr_model.cost_history)]
}
performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv('model_performance.csv', index=False)
print("Saved performance metrics to: model_performance.csv")

# 4. Save cost history for analysis
cost_df = pd.DataFrame({
    'iteration': range(len(lr_model.cost_history)),
    'cost': lr_model.cost_history
})
cost_df.to_csv('training_cost_history.csv', index=False)
print("Saved training history to: training_cost_history.csv")

# 5. Save betting simulation results
betting_results = []
for threshold in [0.55, 0.60, 0.65, 0.70]:
    mask = (test_probabilities > threshold) | (test_probabilities < (1 - threshold))
    if mask.sum() > 0:
        subset_accuracy = accuracy_score(y_test[mask], test_predictions[mask])
        profit_simulation = ((y_test[mask] == test_predictions[mask]).sum() * 100) - ((y_test[mask] != test_predictions[mask]).sum() * 110)
        betting_results.append({
            'confidence_threshold': threshold,
            'games_bet': mask.sum(),
            'accuracy': subset_accuracy,
            'correct_bets': (y_test[mask] == test_predictions[mask]).sum(),
            'wrong_bets': (y_test[mask] != test_predictions[mask]).sum(),
            'simulated_profit': profit_simulation,
            'roi_percent': (profit_simulation / (mask.sum() * 100)) * 100 if mask.sum() > 0 else 0
        })

betting_df = pd.DataFrame(betting_results)
betting_df.to_csv('betting_simulation.csv', index=False)
print("Saved betting simulation to: betting_simulation.csv")

# Plot cost function convergence
lr_model.plot_cost_history()

# Feature importance analysis (weights)
print("\n=== FEATURE IMPORTANCE (WEIGHTS) ===")
feature_weights = lr_model.weights[1:]  # Exclude bias term
bias_term = lr_model.weights[0]

# Get top 10 most important features (by absolute weight)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'weight': feature_weights,
    'abs_weight': np.abs(feature_weights)
}).sort_values('abs_weight', ascending=False)

print(f"Bias term (θ₀): {bias_term:.4f}")
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10)[['feature', 'weight']].to_string(index=False))

# Probability distribution analysis
print("\n=== PROBABILITY ANALYSIS ===")
print(f"Average predicted probability: {test_probabilities.mean():.3f}")
print(f"Probability std: {test_probabilities.std():.3f}")
print(f"Min probability: {test_probabilities.min():.3f}")
print(f"Max probability: {test_probabilities.max():.3f}")

# Compare with different thresholds (ROC analysis mentioned in slides)
print("\n=== THRESHOLD ANALYSIS ===")
thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
for threshold in thresholds:
    predictions_thresh = lr_model.predict(X_test_np, threshold=threshold)
    accuracy_thresh = accuracy_score(y_test, predictions_thresh)
    print(f"Threshold {threshold}: Accuracy = {accuracy_thresh:.4f}")

print("\n=== IMPLEMENTATION NOTES ===")
print("Used sigmoid function g(s) = 1/(1 + exp(-s)) (from class lecture)")
print("Implemented gradient ascent for maximizing log-likelihood (from clas lectures)")
print("Added bias term (θ₀) (from class lectures) ")
print("Used convergence criterion ||θ^{t+1} - θ^t|| < ε (from class lectures)")
print("Handled the concave optimization problem (from class lectures)")
print("No closed-form solution, used iterative method as expected (from class lectures)")


print("\n=== OUTPUT FILES CREATED ===")
print("Analysis Files:")
print("  • logistic_regression_predictions.csv - All game predictions with probabilities")
print("  • feature_importance.csv - Which NFL stats matter most")
print("  • model_performance.csv - Accuracy and key metrics")
print("  • training_cost_history.csv - Algorithm convergence data")
print("  • betting_simulation.csv - Profit/loss for different strategies")

print("\n Visualization:")
print("  • Cost function convergence plot (displayed)")

print("\n=== NEXT STEPS ===")
print("1. Open the CSV files to analyze results in detail")
print("2. Use feature_importance.csv to understand what predicts wins")
print("3. Use betting_simulation.csv to plan future 2025 strategy")
print("4. Compare different confidence thresholds for optimal ROI")

# VISUALIZATIONS
if PLOTTING_AVAILABLE:
    print("\n=== CREATING ADVANCED VISUALIZATIONS ===")
    
    # Set style for professional plots
    try:
        import seaborn as sns
        sns.set_palette("husl")
        SEABORN_AVAILABLE = True
    except ImportError:
        SEABORN_AVAILABLE = False
    
    # 1. FEATURE IMPORTANCE VISUALIZATION
    print("Creating feature importance plots...")
    
    # Categorize features
    def categorize_feature(feature_name):
        if feature_name == 'bias_term':
            return 'Model Bias'
        elif 'rolling' in feature_name:
            return 'Rolling Performance'
        elif any(stat in feature_name for stat in ['point_diff', 'advantage', 'off_vs']):
            return 'Season Statistics'
        elif feature_name.startswith(('home_', 'away_')) and len(feature_name.split('_')) == 2:
            return 'Team Identity'
        else:
            return 'Other'
    
    feature_importance_df['category'] = feature_importance_df['feature'].apply(categorize_feature)
    
    # Top 15 most important features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(15)
    
    category_colors = {
        'Team Identity': '#FF6B6B',
        'Rolling Performance': '#4ECDC4', 
        'Season Statistics': '#45B7D1',
        'Model Bias': '#96CEB4',
        'Other': '#FECA57'
    }
    
    colors = [category_colors.get(cat, '#95A5A6') for cat in top_features['category']]
    
    bars = plt.barh(range(len(top_features)), top_features['abs_weight'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
    plt.xlabel('Absolute Weight (Importance)', fontsize=12, fontweight='bold')
    plt.title('Top 15 Most Important Features for NFL Game Prediction', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, weight) in enumerate(zip(bars, top_features['abs_weight'])):
        plt.text(weight + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{weight:.3f}', va='center', fontsize=9)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=category) 
                      for category, color in category_colors.items() 
                      if category in top_features['category'].values]
    plt.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True)
    
    plt.tight_layout()
    plt.savefig('feature_importance_top15.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)
    
    # 2. ROI ANALYSIS VISUALIZATION
    print("Creating ROI analysis plots...")
    
    plt.figure(figsize=(14, 6))
    
    # Left subplot: ROI by confidence threshold
    plt.subplot(1, 2, 1)
    roi_data = betting_df.sort_values('confidence_threshold')
    
    bars = plt.bar(range(len(roi_data)), roi_data['roi_percent'], 
                   color=['#FF6B6B', '#FFA726', '#66BB6A', '#42A5F5'])
    plt.xticks(range(len(roi_data)), [f"{t:.0%}" for t in roi_data['confidence_threshold']])
    plt.xlabel('Confidence Threshold', fontweight='bold')
    plt.ylabel('ROI Percentage (%)', fontweight='bold')
    plt.title('ROI by Confidence Threshold', fontweight='bold')
    
    # Add value labels
    for bar, roi in zip(bars, roi_data['roi_percent']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{roi:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Right subplot: Risk vs Reward
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(roi_data['games_bet'], roi_data['roi_percent'], 
                         s=roi_data['accuracy']*500, 
                         c=roi_data['confidence_threshold'], 
                         cmap='viridis', alpha=0.7)
    
    plt.xlabel('Number of Games Bet', fontweight='bold')
    plt.ylabel('ROI Percentage (%)', fontweight='bold')
    plt.title('Risk vs Reward Analysis', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Confidence Threshold', fontweight='bold')
    
    # Add annotations
    for i, row in roi_data.iterrows():
        plt.annotate(f"{row['accuracy']:.0%}", 
                    (row['games_bet'], row['roi_percent']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('roi_analysis.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)
    
    # 3. PREDICTION CONFIDENCE DISTRIBUTION
    print("Creating prediction confidence analysis...")
    
    plt.figure(figsize=(12, 8))
    
    # Probability distribution
    plt.subplot(2, 2, 1)
    plt.hist(test_probabilities, bins=20, alpha=0.7, color='#4ECDC4', edgecolor='black')
    plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    
    # Accuracy by confidence level
    plt.subplot(2, 2, 2)
    confidence_levels = ['LOW', 'MEDIUM', 'HIGH']
    conf_accuracy = []
    conf_counts = []
    
    for level in confidence_levels:
        mask = results_df['confidence_level'] == level
        if mask.sum() > 0:
            accuracy = results_df[mask]['correct_prediction'].mean()
            count = mask.sum()
            conf_accuracy.append(accuracy)
            conf_counts.append(count)
        else:
            conf_accuracy.append(0)
            conf_counts.append(0)
    
    bars = plt.bar(confidence_levels, conf_accuracy, 
                   color=['#FF6B6B', '#FFA726', '#66BB6A'])
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Confidence Level')
    plt.ylim(0, 1)
    
    # Add count labels
    for bar, acc, count in zip(bars, conf_accuracy, conf_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{acc:.1%}\n({count} games)', ha='center', va='bottom')
    
    # Profit by confidence
    plt.subplot(2, 2, 3)
    profit_by_conf = []
    for level in confidence_levels:
        mask = results_df['confidence_level'] == level
        if mask.sum() > 0:
            correct = results_df[mask]['correct_prediction'].sum()
            wrong = mask.sum() - correct
            profit = (correct * 100) - (wrong * 110)
            profit_by_conf.append(profit)
        else:
            profit_by_conf.append(0)
    
    colors = ['red' if p < 0 else 'green' for p in profit_by_conf]
    bars = plt.bar(confidence_levels, profit_by_conf, color=colors, alpha=0.7)
    plt.ylabel('Simulated Profit ($)')
    plt.title('Profit by Confidence Level')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # Timeline analysis
    plt.subplot(2, 2, 4)
    results_df['game_number'] = range(len(results_df))
    results_df['cumulative_profit'] = np.cumsum(
        results_df['correct_prediction'] * 100 - (1 - results_df['correct_prediction']) * 110
    )
    
    plt.plot(results_df['game_number'], results_df['cumulative_profit'], 
             color='#4ECDC4', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Game Number')
    plt.ylabel('Cumulative Profit ($)')
    plt.title('Profit Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)
    
    print("\nVISUALIZATIONS CREATED:")
    print("  • feature_importance_top15.png")
    print("  • roi_analysis.png") 
    print("  • prediction_analysis.png")
    
    print("\nKEY INSIGHTS:")
    print(f"Most important feature: {feature_importance_df.iloc[1]['feature']}")
    best_roi = betting_df.loc[betting_df['roi_percent'].idxmax()]
    print(f"Best strategy: {best_roi['confidence_threshold']:.0%} threshold = {best_roi['roi_percent']:.1f}% ROI")

else:
    print("\nInstall matplotlib for visualizations: pip install matplotlib")