# nfl_random_forest.py
# NFL Random Forest Implementation
# Team: Eshaan, Thavaisya, Dishi, Vivek

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# load the preprocessed data
X = pd.read_csv('betting_features_X.csv')
y = pd.read_csv('betting_targets_y.csv')['home_win'].values

print(f"Dataset: {X.shape[0]} games, {X.shape[1]} features")

# split date wise to prevent data leakage
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# RF configuration
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# this is to train the model
rf_model.fit(X_train, y_train)

# this makes predictions
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# to evaluate the performace
train_acc = rf_model.score(X_train, y_train)
test_acc = rf_model.score(X_test, y_test)

print(f"Training accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")

# feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Most Important Features:")
for i, row in feature_importance.head(5).iterrows():
    print(f"{row['feature']:30s} {row['importance']:.3f}")

# betting simulation - implemented same as logistic regression so easy to compare
def betting_simulation(y_true, y_prob, thresholds=[0.55, 0.60, 0.65, 0.70]):
  
    # we have to simulate betting stratergy with different confidence thresholds
    # to test we will place and win $100 on correct bets, but if we loose then  $110 gone as it is a lost bet
  
    results = []
    
    for threshold in thresholds:
        # we will ony bet if model is confident (high or low probability)
        confident_mask = (y_prob > threshold) | (y_prob < (1 - threshold))
        
        if confident_mask.sum() > 0:
            confident_preds = (y_prob[confident_mask] > 0.5).astype(int)
            confident_actual = y_true[confident_mask]
            
            # calculate accuracy here for this threshold
            correct_bets = (confident_actual == confident_preds).sum()
            total_bets = confident_mask.sum()
            accuracy = correct_bets / total_bets
            
            # now we calculating profit/loss
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

# now simulate the betting scenario
betting_results = betting_simulation(y_test, y_prob)

print(f"\nBetting Results:")
for _, row in betting_results.iterrows():
    print(f"{row['threshold']:.0%} threshold: {row['games_bet']:3.0f} games, "
          f"{row['accuracy']:.1%} accuracy, ${row['profit']:5.0f} profit, "
          f"{row['roi_percent']:5.1f}% ROI")

print(f"Best ROI: {betting_results['roi_percent'].max():.1f}%")

# calculate on of the metrics: Brier Score
def calculate_brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

rf_brier_score = calculate_brier_score(y_test, y_prob)
print(f"Brier Score: {rf_brier_score:.3f}")


# save game prediction results
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'probability': y_prob,
    'correct': (y_test == y_pred).astype(int)
})
results_df.to_csv('rf_predictions.csv', index=False)

feature_importance.to_csv('rf_feature_importance.csv', index=False)
betting_results.to_csv('rf_betting_simulation.csv', index=False)

# create summary report
performance_metrics = pd.DataFrame({
    'metric': ['train_accuracy', 'test_accuracy', 'total_games', 'correct_predictions',
               'best_roi', 'best_threshold', 'feature_count', 'brier_score'],
    'value': [train_acc, test_acc, len(y_test), (y_test == y_pred).sum(),
              betting_results['roi_percent'].max(), 
              betting_results.loc[betting_results['roi_percent'].idxmax(), 'threshold'],
              X.shape[1], rf_brier_score]
})
performance_metrics.to_csv('rf_performance_metrics.csv', index=False)


print(f"\nRF is complete - {test_acc:.1%} accuracy, {betting_results['roi_percent'].max():.1f}% best ROI")
