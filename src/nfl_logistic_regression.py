import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("loading data...")

# check if we have the packages we need
try:
    print("pandas:", pd.__version__)
    print("numpy:", np.__version__)
except ImportError as e:
    print(f"missing package: {e}")
    exit(1)

# Calculate accuracy based on what predictions are correct
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

#confusion matrix to measure how well the model is performing (for evaluation)
def confusion_matrix(y_true, y_pred):
    
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    
    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    
    return matrix

# creates a comparison on model prediction of home wins vs away wins (for evaluation)
def classification_report(y_true, y_pred, target_names=None):
    if target_names is None:
        target_names = ['Away Win', 'Home Win']
    
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    report = "\n"
    report += f"{'':>12} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}\n"
    report += "\n"
    
    for i, label in enumerate(unique_labels):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true == label)
        
        name = target_names[i] if i < len(target_names) else f'Class {label}'
        report += f"{name:>12} {precision:>9.2f} {recall:>9.2f} {f1:>9.2f} {support:>9}\n"
    
    return report

# logistic regression implementation
class LogisticRegression:
    
    # sets up the the model with parameters
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.cost_history = []

    # to convert the output into probability  
    def sigmoid(self, z):
        # prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    # log-likelyhood function, find coefficients that minimize cost
    def compute_cost(self, X, y):
        m = X.shape[0]
        linear_pred = X.dot(self.weights)
        predictions = self.sigmoid(linear_pred)
        
        # avoid log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return cost
    
    # how to adjust the coeffients
    def compute_gradient(self, X, y):
        m = X.shape[0]
        linear_pred = X.dot(self.weights)
        predictions = self.sigmoid(linear_pred)
        gradient = X.T.dot(y - predictions) / m
        return gradient
    
    # adjust coefficient until convergence or iterations
    def fit(self, X, y):
        # add bias term
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        # initialize weights
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features)
        
        # training loop
        for i in range(self.max_iterations):
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            gradient = self.compute_gradient(X, y)
            new_weights = self.weights + self.learning_rate * gradient
            
            # check convergence
            if np.linalg.norm(new_weights - self.weights) < self.tolerance:
                print(f"converged after {i+1} iterations")
                break
                
            self.weights = new_weights
            
            if (i + 1) % 100 == 0:
                print(f"iteration {i+1}, cost: {cost:.6f}")
    
    #probability that team wins every home game
    def predict_proba(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        linear_pred = X.dot(self.weights)
        probabilities = self.sigmoid(linear_pred)
        return probabilities
    
    # convert to binary
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)



#Print result and performance

# load the data
print("loading NFL betting data...")

X = pd.read_csv('betting_features_X.csv')
y = pd.read_csv('betting_targets_y.csv')['home_win'].values

print(f"dataset: {X.shape[0]} games, {X.shape[1]} features")
print(f"home win rate: {y.mean():.1%}")

# split data chronologically - important for time series
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"training: {X_train.shape[0]} games")
print(f"testing: {X_test.shape[0]} games")

# convert to numpy
X_train_np = X_train.values
X_test_np = X_test.values

# train model
print("\ntraining logistic regression...")
model = LogisticRegression(
    learning_rate=0.1,
    max_iterations=1000,
    tolerance=1e-6
)

model.fit(X_train_np, y_train)

# make predictions
print("\ngetting predictions...")
train_preds = model.predict(X_train_np)
test_preds = model.predict(X_test_np)

train_probs = model.predict_proba(X_train_np)
test_probs = model.predict_proba(X_test_np)

# evaluate
train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print(f"\ntrain accuracy: {train_acc:.4f}")
print(f"test accuracy: {test_acc:.4f}")

# detailed results
print(classification_report(y_test, test_preds))

print("\nconfusion matrix:")
cm = confusion_matrix(y_test, test_preds)
print(cm)

# save results
print("\nsaving results...")

# predictions with confidence levels
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': test_preds,
    'probability': test_probs,
    'correct': (y_test == test_preds).astype(int),
    'confidence': ['HIGH' if p > 0.65 or p < 0.35 else 'MEDIUM' if p > 0.55 or p < 0.45 else 'LOW' 
                  for p in test_probs]
})
results_df.to_csv('logistic_regression_predictions.csv', index=False)

# feature importance (weights)
feature_importance = pd.DataFrame({
    'feature': ['bias'] + list(X.columns),
    'weight': model.weights,
    'abs_weight': np.abs(model.weights)
}).sort_values('abs_weight', ascending=False)
feature_importance.to_csv('feature_importance.csv', index=False)

# performance metrics
metrics = {
    'metric': ['train_acc', 'test_acc', 'total_games', 'correct_preds', 
               'home_win_rate', 'bias_term', 'iterations'],
    'value': [train_acc, test_acc, len(y_test), sum(y_test == test_preds),
              y.mean(), model.weights[0], len(model.cost_history)]
}
pd.DataFrame(metrics).to_csv('model_performance.csv', index=False)

# training history
cost_df = pd.DataFrame({
    'iteration': range(len(model.cost_history)),
    'cost': model.cost_history
})
cost_df.to_csv('training_cost_history.csv', index=False)

# betting simulation
betting_results = []
for threshold in [0.55, 0.60, 0.65, 0.70]:
    confident_mask = (test_probs > threshold) | (test_probs < (1 - threshold))
    if confident_mask.sum() > 0:
        subset_acc = accuracy_score(y_test[confident_mask], test_preds[confident_mask])
        # simple profit calc: win $100, lose $110
        wins = (y_test[confident_mask] == test_preds[confident_mask]).sum()
        losses = (y_test[confident_mask] != test_preds[confident_mask]).sum()
        profit = wins * 100 - losses * 110
        
        betting_results.append({
            'confidence_threshold': threshold,
            'games_bet': confident_mask.sum(),
            'accuracy': subset_acc,
            'correct_bets': wins,
            'wrong_bets': losses,
            'profit': profit,
            'roi_percent': (profit / (confident_mask.sum() * 100)) * 100 if confident_mask.sum() > 0 else 0
        })

pd.DataFrame(betting_results).to_csv('betting_simulation.csv', index=False)

# analyze feature importance
print("\nfeature importance (top 10):")
print(feature_importance.head(10)[['feature', 'weight']].to_string(index=False))

print(f"\nbias term: {model.weights[0]:.4f}")

# probability analysis
print(f"\naverage predicted prob: {test_probs.mean():.3f}")
print(f"prob std: {test_probs.std():.3f}")
print(f"min prob: {test_probs.min():.3f}")
print(f"max prob: {test_probs.max():.3f}")

# threshold analysis
print("\nthreshold analysis:")
for thresh in [0.4, 0.45, 0.5, 0.55, 0.6]:
    thresh_preds = model.predict(X_test_np, threshold=thresh)
    thresh_acc = accuracy_score(y_test, thresh_preds)
    print(f"threshold {thresh}: accuracy = {thresh_acc:.4f}")

print("\nfiles saved:")
print("- logistic_regression_predictions.csv")
print("- feature_importance.csv") 
print("- model_performance.csv")
print("- training_cost_history.csv")
print("- betting_simulation.csv")

print(f"\nmodel summary:")
print(f"- {test_acc:.1%} accuracy on {len(y_test)} test games")
print(f"- best roi: {max([r['roi_percent'] for r in betting_results]):.1f}% at high confidence")
print(f"- converged in {len(model.cost_history)} iterations")
print("\ndone!")
