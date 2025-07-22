import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Load data
X = pd.read_csv('betting_features_X.csv')
y = pd.read_csv('betting_targets_y.csv')['home_win'].values

print(f"Dataset: {X.shape[0]} games, {X.shape[1]} features")

# train/test split the data wise
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_train_np = X_train.values
X_test_np = X_test.values

# establish a baseline (liner svm) before hyperparameter optimization
linear_svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
linear_svm.fit(X_train_np, y_train)
linear_pred = linear_svm.predict(X_test_np)
linear_prob = linear_svm.predict_proba(X_test_np)[:, 1]
linear_accuracy = (linear_pred == y_test).mean()

print(f"Linear SVM accuracy: {linear_accuracy:.3f}")

# RBF SVM to test if non linear exists
rbf_svm = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
rbf_svm.fit(X_train_np, y_train)
rbf_pred = rbf_svm.predict(X_test_np)
rbf_prob = rbf_svm.predict_proba(X_test_np)[:, 1]
rbf_accuracy = (rbf_pred == y_test).mean()

print(f"RBF SVM accuracy: {rbf_accuracy:.3f}")

# Hyperparameter tuning, test diff svm configurations
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 0.01, 0.1]
}

# time series cross validation
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

# optimization process
grid_search.fit(X_train_np, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.3f}")

# get the final model
best_svm = grid_search.best_estimator_
final_pred = best_svm.predict(X_test_np)
final_prob = best_svm.predict_proba(X_test_np)[:, 1]
final_accuracy = (final_pred == y_test).mean()

print(f"Test accuracy: {final_accuracy:.3f}")

# betting simulation
def betting_simulation(y_true, y_prob, thresholds=[0.55, 0.60, 0.65, 0.70]):
    results = []
    
    for threshold in thresholds:
        confident_mask = (y_prob > threshold) | (y_prob < (1 - threshold))
        
        if confident_mask.sum() > 0:
            confident_preds = (y_prob[confident_mask] > 0.5).astype(int)
            confident_actual = y_true[confident_mask]
            
            correct_bets = (confident_actual == confident_preds).sum()
            total_bets = confident_mask.sum()
            accuracy = correct_bets / total_bets
            
            wrong_bets = total_bets - correct_bets
            profit = correct_bets * 100 - wrong_bets * 110
            roi = (profit / (total_bets * 100)) * 100
            
            results.append({
                'threshold': threshold,
                'games_bet': total_bets,
                'accuracy': accuracy,
                'correct_bets': correct_bets,
                'wrong_bets': wrong_bets,
                'profit': profit,
                'roi_percent': roi
            })
    
    return pd.DataFrame(results)

betting_results = betting_simulation(y_test, final_prob)

print(f"\nBetting Results:")
for _, row in betting_results.iterrows():
    print(f"{row['threshold']:.0%} threshold: {row['games_bet']:3.0f} games, "
          f"{row['accuracy']:.1%} accuracy, ${row['profit']:5.0f} profit, "
          f"{row['roi_percent']:5.1f}% ROI")

print(f"Best ROI: {betting_results['roi_percent'].max():.1f}%")

# brier Score
def calculate_brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

svm_brier_score = calculate_brier_score(y_test, final_prob)
print(f"Brier Score: {svm_brier_score:.3f}")

# main predictions file
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': final_pred,
    'probability': final_prob,
    'correct': (y_test == final_pred).astype(int),
    'confidence': ['HIGH' if p > 0.65 or p < 0.35 else 'MEDIUM' if p > 0.55 or p < 0.45 else 'LOW' 
                  for p in final_prob]
})
results_df.to_csv('svm_predictions.csv', index=False)

# betting results
betting_results.to_csv('svm_betting_simulation.csv', index=False)

# overall performance metrics with brier score
performance_metrics = pd.DataFrame({
    'metric': ['test_accuracy', 'total_games', 'correct_predictions',
               'best_roi', 'best_threshold', 'support_vectors', 'kernel_type', 'brier_score'],
    'value': [final_accuracy, len(y_test), (y_test == final_pred).sum(),
              betting_results['roi_percent'].max(), 
              betting_results.loc[betting_results['roi_percent'].idxmax(), 'threshold'],
              sum(best_svm.n_support_), best_svm.kernel, svm_brier_score]
})
performance_metrics.to_csv('svm_performance_metrics.csv', index=False)

# best hyperparameters that we find
tuning_results = pd.DataFrame({
    'parameter': list(grid_search.best_params_.keys()),
    'best_value': list(grid_search.best_params_.values())
})
tuning_results.to_csv('svm_hyperparameters.csv', index=False)

# create comparison table for final conclusion part
model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'SVM'],
    'Test_Accuracy': [0.645, 0.684, final_accuracy],
    'Best_ROI': [41.7, 45.2, betting_results['roi_percent'].max()],
    'Key_Strength': ['Interpretable', 'Feature interactions', 'Maximum margin'],
    'Complexity': ['Low', 'Medium', 'Medium']
})
model_comparison.to_csv('model_comparison_final.csv', index=False)


print(f"\nSVM complete - {final_accuracy:.1%} accuracy, {betting_results['roi_percent'].max():.1f}% best ROI")
