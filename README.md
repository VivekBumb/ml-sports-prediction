# Proposal


From the National Football League to the Ultimate Fighting Championship, sports betting has become increasingly popular amongst American youth. One survey by Statista showed that 30% of respondents had put money on sports with the NFL being most popular[1].


Sports betting can be seen as a predatory industry, as it is addictive and almost all people lose money. The spreads and betting odds are largely manipulated by social media trends and analysis of how much money is being placed on each team. A study on German soccer showed [2] that significant portions of betting odds were based on perceived team momentum, which has no correlation with actual success. This ensures companies like DraftKings and FanDuel can make the largest profit possible. Therefore, constructing a model using only objective metrics like player stats should be more accurate and enable consumers to make a profit.


Our analysis utilizes NFL team offensive and defensive statistics spanning from 2020 to 2024 [3]. The dataset includes season-end averages for key metrics such as points scored, total yards, plays run, yards per play, turnovers, first downs, and comprehensive passing statistics including completions, attempts, passing yards, touchdowns, and interceptions. This 5-year historical dataset provides extensive training data across recent seasons, ensuring relevance to modern NFL gameplay and strategy.


To prepare data, missing numeric values will be imputed using median to account for outliers. Duplicates will be removed to ensure each entry is unique. We will engineer features including 5-game rolling averages of point differentials and yards/play, plus relative performance metrics capturing team strengths and weaknesses relative to opponents. One-hot coding will convert team names into numerical values for algorithm compatibility. Since algorithms like logistic regression use gradient-based optimization, features with huge scales cause erratic training. Therefore, continuous features will be standardized [(original_value - mean) / SD] to give each feature equal importance [4].


The main supervised learning algorithms we are implementing are logistic regression, random forest, and support vector machines. Logistic regression serves as baseline, fitting weighted sums of input features to estimate win probabilities. Coefficient weights establish how each statistic drives game outcomes, especially helpful during initial modeling stages. Random forest determines nonlinear interactions by averaging votes of many decision trees, providing robust probability estimates and built-in feature ranking. SVM finds the optimal decision boundary separating wins from losses, predicting which side new games will fall under [5].


We will evaluate our model using three quantitative metrics. First, prediction accuracy will be determined by calculating the percentage of correct game outcome predictions across test games against sportsbook odds and spreads. Then, Return on Investment (ROI) will measure profitability by simulating bets placed when our model's prediction probability exceeds implied probability from sportsbook odds [6]. The final metric will be Brier score, which evaluates probabilistic prediction quality by calculating how close predicted probabilities are to actual outcomes [7].


Our primary goal is achieving 55-60% prediction accuracy based on existing NFL prediction research [8], while maintaining positive ROI over multiple seasons. This target exceeds the 52.4% threshold needed to overcome typical -110 betting odds. From an ethical standpoint, we aim to create a statistic-driven alternative to perception-manipulated betting lines. Regarding sustainability, our model emphasizes statistical validity over short-term gains, encouraging disciplined approaches to sports betting.

#

| Name | Proposal Contributions |
|------|----------------------|
| Eshaan | Intro, background, problem |
| Thavaisya | Methods (ML algorithms) |
| Dishi | Methods (Preprocessing) and quantitative metrics |
| Vivek | Dataset info, Potential results and discussion |

# Data Collection Process

## Overview

Our NFL betting prediction system uses the `nfl_data_py` package to collect official NFL data directly from league sources, eliminating the need for web scraping and ensuring reliable, consistent data quality.

## Data Sources

```python
# Primary data collection via nfl_data_py package
import nfl_data_py as nfl

# Two main data streams:
games = nfl.import_schedules([2020, 2021, 2022, 2023, 2024])       # Game results
weekly_stats = nfl.import_weekly_data([2020, 2021, 2022, 2023, 2024])  # Team stats
```

## Data Coverage

- **Time Period:** 2020-2024 NFL seasons (5 complete seasons)
- **Game Data:** 1,408 total games including playoffs
- **Team Statistics:** Weekly performance metrics for all 32 NFL teams
- **Data Quality:** Official NFL statistics (no web scraping required)

## Key Advantages

**Official Data:** Direct from NFL sources via nfl_data_py (no web scraping) 
**Consistent Format:** Standardized column names and data types  
**Real-time Updates:** Package automatically handles data formatting changes  
**Complete Coverage:** All games, teams, and seasons in one source  

## Data Download Script

```bash
# Run the data collection and preprocessing
python simplified_nfl_data.py
```
Note: Command could be python3 or depending upon the python version installed on the system.

## Dependencies

```bash
pip install nfl_data_py pandas numpy
```

## Output Files Generated by simplified_nfl_data.py

- `clean_nfl_games.csv` - Complete game results with scores and metadata

## Error Handling

The script includes robust error handling for:

- Missing `nfl_data_py` package installation
- Data compatibility checks



## Data Freshness

- **Current dataset:** Through 2024 NFL season (complete)
- **For 2025 predictions:** Script can be easily updated to include new season data
- **Update frequency:** Run script after each NFL season concludes


# Data Preprocessing

## Overview

The preprocessing transforms raw NFL game data into ML ready features through statistical aggregation, feature engineering, and standardization. This follows the methodology outlined in our research proposal for optimal predictive performance.

### 1. Data Cleaning & Validation
**Functions:** `.dropna()`, `.astype()`

Raw NFL data contains incomplete games with missing scores due to postponements or data collection issues. We filtered these using `.dropna()` which removed games without complete scoring information, retaining 1,408 complete games from 2020-2024 seasons.
We also converted game outcomes to binary targets using `.astype(int)`, creating home team win/loss indicators (1/0) required for logistic regression. This ensures our dataset contains only valid, complete games with clear outcomes for reliable model training.


### 2. Season-Level Team Statistics
**Functions:** `.groupby()`, `.agg()`, `.reset_index()`

Season averages establish each team's baseline offensive and defensive capabilities. We aggregated weekly team performance using  `.groupby(['season', team_col])` to calculate mean passing and rushing yards for each team per season. This provides fundamental team strength metrics that capture overall quality and playing style.
The `.agg()` function computes season-long averages while `.reset_index()` restructures the data for easy merging with game-level features. These season statistics serve as the foundation for comparing team strengths in individual matchups.


### 3. Rolling Performance Metrics
**Functions:** `.iloc()`, `.mean()`

Recent team performance captures momentum and current form, which is more predictive than season-long averages for sports outcomes. We calculated 5-game rolling windows for each team using `.iloc()` to select the last five games and `.mean()` to compute averages for win percentage, points scored, and point differential.
Critical to data integrity, rolling statistics only use games before the prediction target, preventing future information leakage. Early season games with fewer than 5 prior games use all available games to ensure complete coverage.


**Rolling Features Created:**
- `home_rolling_win_pct` / `away_rolling_win_pct`
- `home_rolling_points` / `away_rolling_points`
- `rolling_win_advantage` / `rolling_point_diff_advantage`

### 4. Matchup-Level Feature Engineering
**Functions:** `Calculations`  
Football outcomes depend on relative team strengths rather than absolute performance. We created direct comparison features like `yards_advantage` (home team total yards - away team total yards), `rolling_win_advantage` (home rolling win % - away rolling win %), and `rolling_point_diff_advantage` (home recent point differential - away recent point differential).
These head-to-head features capture competitive dynamics between specific opponents, providing the model with relative performance metrics that directly predict game outcomes.


### 5. One-Hot Team Encoding
**Functions:** `pd.get_dummies()`, `pd.concat()`

Machine learning algorithms require numerical inputs and cannot process categorical team names like "Chiefs" or "Patriots". We used `pd.get_dummies()` to create binary indicator variables for each of the 32 NFL teams, generating separate columns for home and away team identity (e.g., home_KC, away_NE).
This encoding allows the model to learn team-specific patterns and home field advantages while maintaining the numerical format required for logistic regression training.


### 6. Feature Standardization
**Functions:** `.mean()`, `.std()`

Features with different scales (points vs percentages) would dominate gradient descent optimization and prevent proper algorithm convergence. We standardized all numerical features using the formula (value - mean) / standard_deviation, ensuring each feature has mean ≈ 0 and standard deviation ≈ 1.
This scaling gives equal importance to all features during logistic regression training, preventing large-scale metrics from overshadowing smaller but potentially more predictive variables.


## Feature Categories Created

### Performance Features (12 total): HOW GOOD ARE THE TEAMS?

- **Season Stats:** `home_point_diff`, `away_point_diff`, `point_diff_advantage`
- **Matchup Stats:** `home_off_vs_away_def`, `away_off_vs_home_def`, `yards_advantage`
- **Rolling Stats:** `home_rolling_win_pct`, `rolling_win_advantage`, `rolling_point_diff_advantage`

### Team Identity Features (64 total): WHO IS PLAYING?

- **Home Teams:** `home_ARI`, `home_ATL`, `home_BAL`, ..., `home_WAS` (32 features)
- **Away Teams:** `away_ARI`, `away_ATL`, `away_BAL`, ..., `away_WAS` (32 features)

## Output Specifications
Our preprocessing pipeline transforms raw NFL game data into a machine learning-ready dataset optimized for binary classification. The final dataset contains 1,408 NFL games with 76 engineered features capturing team performance and matchup dynamics.
### Final Dataset Dimensions

- **Samples:** 1,408 NFL games (2020-2024)
- **Features:** 76 total features
  - 12 performance/statistical features
  - 64 team encoding features (32 home + 32 away)
- **Target:** Binary home team victory (0/1)

### Feature Scaling Results
Standardization ensures equal feature contribution during logistic regression training.
- **Numerical features:** Mean ≈ 0, Standard deviation ≈ 1
- **Binary features:** Values in {0, 1}
- **No missing values:** Complete dataset ready for ML

This setup allows the model to fairly compare all features and clearly show which NFL statistics are most important for predicting wins.

## Output Files Generated by simplified_nfl_data.py

- `team_season_stats.csv` - Aggregated team performance by season
- `betting_features_X.csv` - ML-ready feature matrix (1,408 × 76)
- `betting_targets_y.csv` - Binary target variable (home team wins)
- `complete_betting_dataset.csv` - Full dataset with all features and targets
- `feature_names.txt` - List of all 76 feature names


# Model Implementation/Evaluation - Logistic Regression
Our approach consists of four primary phases: data acquisition, preprocessing, feature engineering, and model training. The following section will be discussing the training for our first model: logistic regression, implemented in `nfl_logistic_regression.py`.

```bash
# Run the logistic regression algorithm
python nfl_logistic_regression.py
```

**Justifications:** We chose logistic regression as our baseline model because it's easy to understand and gives probability outputs needed for betting decisions. The algorithm shows clear coefficient weights that tell us which NFL statistics are most important for predicting wins.
Key advantages for our project:
-	Clear coefficients show which features matter most
-	Probability outputs help decide betting confidence levels
-	Fast training allows quick updates with new game data
-	Baseline performance gives us a starting point to compare other models  

We split the data chronologically - first 80% of games (1,126) for training and last 20% (282) for testing. This prevents data leakage and mimics real betting where we predict future games using only past data.


## Logistic Regression Steps

## 1. Data Preparation
**Functions:** `pd.read_csv()`  
Loads our preprocessed NFL dataset containing 1,408 games with 76 features and binary home win targets. The data is already cleaned and standardized from our preprocessing pipeline.

## 2. Model Initialization
**Functions:** `__init__()`  
Sets up the logistic regression model with learning parameters: learning rate (0.1), maximum iterations (1,000), and convergence tolerance. These control how fast the model learns and when to stop training.

## 3. Start Training 
**Functions:** `fit()`  
Begins the training process by adding a bias term and initializing 77 random weights (one for each feature plus bias). The model starts with random guesses for all coefficients.

## 4. Training Loop (1000 Iterations)
**Functions:** `for loop` inside `fit()`  
Repeats the learning process up to 1,000 times. In each iteration, the model makes predictions on all training games and adjusts weights to reduce prediction errors.

### 4a: Calculate Cost Function
**Functions:** `compute_cost()`   
Measures how wrong the current predictions are by comparing them to actual game results across all 1,126 training games.

#### 4a.1: Apply Sigmoid Function
**Functions:** `sigmoid()`  
Converts raw mathematical calculations into probabilities between 0 and 1, representing the chance of home team victory.

### 4b: Compute Gradient
**Functions:** `compute_gradient()`  
Determines which direction to adjust each of the 77 coefficients to improve predictions.

### 4c: Weight Update
**Functions:** Direct assignment with `+`, `*` operators
Updates all coefficients based on gradient recommendations, making small improvements to prediction accuracy.

### 4d: Check Convergence
**Functions:** `np.linalg.norm()`  
Checks if coefficient changes are very small, indicating the model has learned as much as possible and training can stop.

## 5: Store Optimized Coefficients
**Functions:** Assignment to `self.weights`  
Assignment to self.weights
After training completes, saves the final 77 optimized coefficients that predict NFL games with 66.3% accuracy.

### Quantitative Metrics
We evaluated the model using 3 key metrics:
- **Accuracy**: Proportion of correctly predicted outcomes
- **Return on Investment (ROI)**: Simulated betting performance using confidence threshold of 60%
- **Brier Score**: Level of predictive accuracy

## Output Files Generated by nfl_logistic_regression.py

- `logistic_regression_predictions.csv` - All game predictions with probabilities and confidence levels
- `feature_importance.csv` - Ranked coefficients showing which NFL stats matter most for predictions
- `model_performance.csv` - Accuracy metrics and key performance indicators
- `training_cost_history.csv` - Algorithm convergence data from gradient ascent
- `betting_simulation.csv` - ROI analysis for different confidence thresholds


## Results/Discussion

### Test Accuracy  
- Achieved **66.3%**, which is significantly above the 52.4% break-even threshold for betting  
- Home win rate baseline: 53.8%  
- Indicates that the model captures meaningful structure in NFL outcomes

---

### Return on Investment (ROI) Simulation

We simulated a betting strategy that places a bet only when the predicted win probability exceeded 60% (or was below 40% for away wins). Bets are $100 each, with +100 payout and -110 loss (standard sportsbook odds).

- **Games Bet:** 173  
- **Correct Bets:** 122  
- **Incorrect Bets:** 51  
- **Total Profit:** **$6,590.00**  
- **Total Wagered:** $17,300.00  
- **ROI:** **38.1%**

Model confidence was highly predictive of betting success — the high-confidence bets had over 72.7% accuracy.

---

### Brier Score – Probability Calibration

To measure how well our predicted probabilities aligned with actual outcomes, we evaluated the Brier score.

- **Brier Score:** **0.219**

Interpretation:
- A perfect model = 0.0  
- A naive model (50/50 every time) = 0.25  
- Our model = 0.219 → indicates a good level of predictive accuracy

---

### Visual Analysis

#### Cost Function Convergence

![Cost Function Convergence](results/visualizations/cost_convergence_plot.png)

- Gradient ascent trained for full 1000 iterations
- Cost plot showed steady decline (no oscillation)

#### Feature Importance

![Feature Importance](results/visualizations/feature_importance_top15.png)

- Top 5 features included:
  - `away_KC (-0.593, strongest predictor)`
  - `home_ARI (-0.527)`
  - `home_KC (+0.465)`
  - `away_JAX (+0.351)`
  - `home_GB (+0.346)`

- Suggests both recent team form and team identity matter a significant amount

#### Prediction Confidence and Profit Timeline

![Prediction Analysis](results/visualizations/prediction_analysis.png)

- Accuracy increased with confidence:
  - Low confidence: ~59.6%  
  - Medium confidence: ~61.2%  
  - High confidence: **72.7%**
 
- Net profit of $6,590 was earned over the 2024 test season (60% confidence threshold)
- All games: $8,250 profit (but higher risk)
- Profit grew steadily, indicating stable model performance over time

#### ROI vs Confidence Threshold
 
![ROI Analysis](results/visualizations/roi_analysis.png)

- ROI peaked when betting only on games with model confidence ≥ 60%
- Most profitable threshold range: **65–70%**
    - 65% threshold: 42.7% ROI ($5,640 profit on 132 games)
    - 70% threshold: 46.3% ROI ($4,170 profit on 90 games)

---


#

| Name | Midterm Contributions |
|------|----------------------|
| Kevin | README documentation, project introduction, problem statement, and final report writing |
| Thavaisya | Data collection/download, results analysis and visualizations |
| Dishi | Data preprocessing pipeline, feature engineering, and performance evaluation metrics |
| Vivek | Model implementation (logistic regression) |


# Model Implementation/Evaluation - Random Forest

Following our successful logistic regression baseline, we implemented Random Forest as our second model to capture nonlinear patterns in NFL game outcomes. This section discusses the training for our Random Forest model, implemented in `nfl_random_forest.py`.

```bash
# Run the Random Forest algorithm
python nfl_random_forest.py
```

**Justifications:** We chose Random Forest as the second model to capture nonlinear patterns and feature interactions that might exist in NFL game data. Random Forest works by training many decision trees and having them vote together on predictions.

Key advantages for this NFL project:
- Can find patterns between features (like when good offense meets weak defense)
- Gets more reliable predictions by having 300 trees vote instead of just one model
- Shows us which NFL stats work best together for predicting games
- Gives us confidence scores for deciding which games to bet on
- Works well with both season-long stats and recent team performance data

We used the same chronological data split as logistic regression - first 80% of games (1,126) for training and last 20% (282) for testing. This ensures fair comparison between models.

## Random Forest Implementation Steps

### 1. Data Preparation
**Functions:** `pd.read_csv()`  
Loads the same preprocessed NFL dataset used for logistic regression: 1,408 games with 76 features. This ensures fair comparison between our models since they train on identical data.

### 2. Model Configuration  
**Functions:** `RandomForestClassifier()`  
Configures the Random Forest model specifically for the NFL dataset:
- **300 trees (n_estimators)**: Creates ensemble for voting on 282 test games
- **log₂(76) ≈ 6 features per split (max_features='log2')**: Randomly selects from NFL features at each split
- **Bootstrap sampling enabled**: Each tree trains on different subsets of 1,126 training games
- **Max depth 10**: Controls complexity to prevent memorizing specific NFL game patterns

### 3. Training Process
**Functions:** `fit()`  
Trains the ensemble on NFL training data. The implementation:
1. Creates 300 decision trees using 1,126 training games
2. Each tree randomly samples from NFL games (with replacement)
3. At each split, randomly selects 6 features from 76 NFL statistics (log₂(76) ≈ 6)
4. Builds decision rules using features like `rolling_win_advantage`, `home_rolling_points`, team encodings
5. Each tree learns different patterns from the preprocessed NFL data

### 4. Ensemble Prediction
**Functions:** `predict()`, `predict_proba()`  
Makes predictions on 282 test games using ensemble voting:
- **Classification**: All 300 trees vote on each NFL game outcome (home win/loss)
- **Probability**: Averages all tree probabilities to get confidence for betting decisions
- **Implementation**: For each test game, combines predictions from all trees trained on NFL data
- **Output**: Generates probabilities for the betting simulation strategy

### 5. Hyperparameter Optimization
**Functions:** `GridSearchCV()`, `TimeSeriesSplit()`  
Optimizes model parameters using the NFL dataset:
- **Number of trees**: Tests 100, 200, 300 trees on training data
- **Features per split**: Experiments with different numbers of the 76 NFL features  
- **Tree complexity**: Adjusts max_depth for specific data patterns
- **Cross-validation**: Uses chronological splits on 2020-2024 NFL games (not random splits)
- **Goal**: Find best configuration for predicting 282 test games

## Output Files Generated by nfl_random_forest.py

- `rf_predictions.csv` - All game predictions with probabilities and confidence levels
- `rf_feature_importance.csv` - Feature rankings from ensemble voting
- `rf_betting_simulation.csv` - ROI analysis for different confidence thresholds  
- `rf_performance_metrics.csv` - Model accuracy and performance summary
- `rf_hyperparameters.csv` - Best parameter settings found
- `rf_feature_importance.png` - Visualization of top important features

## Results/Discussion

### Performance Metrics
- **Test Accuracy:** 68.4% (193 correct out of 282 games)
- **Training Accuracy:** 76.6% (good learning without severe overfitting)

### Betting Simulation Results

| Confidence Threshold | Games Bet | Accuracy | ROI |
|---------------------|-----------|----------|-----|
| 55% | 224 | 68.8% | 34.4% |
| 60% | 167 | 70.7% | 38.4% |
| 65% | 112 | 73.2% | 43.8% |
| **70%** | **69** | **73.9%** | **45.2%** |

**Key Finding:** Higher confidence thresholds produce fewer bets but significantly better ROI and accuracy.

### Feature Importance Discovery

Random Forest identified **matchup analysis** as most predictive for NFL games:

**Top 5 Features:**
1. `home_off_vs_away_def (0.131)` - Home offensive advantage
2. `yards_advantage (0.128)` - Total yards comparison  
3. `rolling_point_diff_advantage (0.116)` - Recent dominance
4. `away_off_vs_home_def (0.109)` - Away offensive advantage
5. `away_rolling_points (0.087)` - Away team recent scoring

**Key Insights:**
- **Matchup features dominate:** Top 4 focus on offensive advantages and yards
- **Recent performance matters:** Rolling averages outweigh season stats
- **Team identity less important:** Specific team features rank lower

### Hyperparameter Optimization
**Optimal Configuration:** 300 trees, max depth 10, log₂(76) features per split, min 15 samples per split

### Visual Analysis

#### Feature Importance Analysis
![Feature Importance](results/visualizations/rf_feature_importance.png)
Shows comprehensive ranking of NFL statistics - matchup and rolling performance features dominate over team identity. Shows the model cares more about "how teams match up against each other" and "recent performance" rather than "which specific teams are playing."

#### Prediction Performance
![Prediction Analysis](results/visualizations/rf_prediction_analysis.png)
Probability distribution and accuracy by confidence level - High confidence predictions achieve 73.2% accuracy (112 games), Medium confidence 64.3% (112 games), Low confidence 67.2% (58 games). Shows the model is well-calibrated - when it's confident about a prediction, it's usually right. This makes it trustworthy for betting decisions.

#### Feature Categories
![Feature Analysis](results/visualizations/rf_feature_analysis.png)
Breakdown by category: Rolling/Momentum (43.9%), Matchup/Season (36.9%), Team Identity (19.2%). Shows recent team form and matchup analysis are far more important than team reputation/identity for predicting NFL games.

## Summary

**Performance:** 68.4% accuracy with 45.2% ROI using selective betting strategy  
**Discovery:** Matchup analysis and recent form more predictive than team identity  
**Strategy:** 70% confidence threshold optimal for profitable NFL betting  
**Model:** 300-tree ensemble effectively captures complex NFL patterns

## Next Steps
- Implement **Support Vector Machine (SVM)** for optimal decision boundary analysis
- Create comprehensive comparison of all three models

## Gantt Chart

![image](https://github.gatech.edu/vbumb3/ml-sports-prediction/blob/main/Gantt_chart.png)

## Directory Structure  
`/`: Root directory of the NFL betting prediction project  
`/README.md`: Main project documentation with methodology, results, and analysis

`/data/`: All datasets and processed data files  
`/data/processed/`: Machine learning ready datasets  
`/data/processed/betting_features_X.csv`: ML-ready feature matrix (1,408 × 76)  
`/data/processed/betting_targets_y.csv`: Binary target variable (home team wins)  
`/data/processed/complete_betting_dataset.csv`: Full dataset with all features and targets  
`/data/processed/clean_nfl_games.csv`: Cleaned NFL game results with scores and metadata  
`/data/processed/team_season_stats.csv`: Aggregated team performance by season  
`/data/processed/feature_names.txt`: List of all 76 feature names  

`/src/`: Source code for data processing and machine learning  
`/src/simplified_nfl_data.py`: Data download and preprocessing pipeline  
`/src/logistic_regression.py`: Logistic regression model implementation and training  

`/results/`: All model outputs and analysis results  
`/results/predictions/`: CSV files with model predictions and performance metrics  
`/results/predictions/logistic_regression_predictions.csv`: All game predictions with probabilities and confidence levels  
`/results/predictions/feature_importance.csv`: Ranked coefficients showing which NFL stats matter most for predictions  
`/results/predictions/betting_simulation.csv`: ROI analysis for different confidence thresholds  
`/results/predictions/model_performance.csv`: Accuracy metrics and key performance indicators  
`/results/predictions/training_cost_history.csv`: Algorithm convergence data from gradient ascent  

`/results/visualizations/`: Charts and plots for analysis  
`/results/visualizations/cost_convergence_plot.png`: Training convergence visualization  
`/results/visualizations/feature_importance_top15.png`: Top 15 most important features chart  
`/results/visualizations/roi_analysis.png`: ROI analysis by confidence threshold  
`/results/visualizations/prediction_analysis.png`: Prediction confidence and profit analysis  


## References:

[1] "Americans Torn About Sports Betting," Statista, 2021. [Online]. Available: https://www.statista.com/chart/26178/sports-betting-attitudes-us/

[2] T. Angelis et al., "Momentum and betting market efficiency in German soccer," Journal of Sports Economics, vol. 23, no. 4, pp. 456-478, 2022.

[3] "NFL Data via nfl_data_py Package," Official NFL Statistics. [Online]. Available: https://pypi.org/project/nfl-data-py/

[4] Phatak, A. A., Mehta, S., Wieland, F.-G., Jamil, M., Connor, M., Bassek, M., & Memmert, D. (2022). Context is key: Normalization as a novel approach to sport specific preprocessing of KPI’s for Match Analysis in soccer. Scientific Reports, 12(1). https://doi.org/10.1038/s41598-022-05089-y 

[5] Kim, C., Park, J.-H., & Lee, J.-Y. (2024). AI-based betting anomaly detection system to ensure fairness in sports and prevent illegal gambling. Scientific Reports, 14(1). https://doi.org/10.1038/s41598-024-57195-8 

[6] Walsh, C., & Joshi, A. (2024). Machine Learning for Sports Betting: Should Model Selection Be Based on Accuracy or Calibration? https://doi.org/10.2139/ssrn.4705918 

[7] Steyerberg, E. W., Vickers, A. J., Cook, N. R., Gerds, T., Gonen, M., Obuchowski, N., Pencina, M. J., & Kattan, M. W. (2010). Assessing the performance of prediction models. Epidemiology, 21(1), 128–138. https://doi.org/10.1097/ede.0b013e3181c30fb2

[8] R. M. Galekwa et al., "A Systematic Review of Machine Learning in Sports Betting: Techniques, Challenges, and Future Directions," arXiv preprint arXiv:2410.21484, 2024.






