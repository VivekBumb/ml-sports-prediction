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

## Preprocessing Steps

### 1. Data Cleaning & Validation
**Functions:** `.dropna()`, `.astype()`

Raw NFL data contains incomplete games with missing essential information that would cause errors in downstream calculations. After cleaning, 1,408 complete games remained for analysis. Machine learning algorithms also require numerical targets (0/1) rather than categorical outcomes.

### 2. Season-Level Team Statistics
**Functions:** `.groupby()`, `.agg()`, `.reset_index()`

Season averages establish each team's baseline strength and overall quality. It provides essential context for predicting individual game matchups.

### 3. Rolling Performance Metrics
**Functions:** `.iloc()`, `.mean()`

Recent team performance (momentum / form) is more predictive than season-long averages for sports outcomes.

**Rolling Features Created:**
- `home_rolling_win_pct` / `away_rolling_win_pct`
- `home_rolling_points` / `away_rolling_points`
- `rolling_win_advantage` / `rolling_point_diff_advantage`

### 4. Matchup-Level Feature Engineering
**Functions:** `Calculations`  
Football outcomes depend more on relative matchups than absolute team strength. A good offense vs weak defense creates different dynamics than good offense vs strong defense.

### 5. One-Hot Team Encoding
**Functions:** `pd.get_dummies()`, `pd.concat()`

Machine learning algorithms require numerical inputs and cannot process categorical text data like team names (e.g., "Chargers", "Dolphins").

### 6. Feature Standardization
**Functions:** `.mean()`, `.std()`

Features with different scales (points vs percentages) would dominate the gradient descent optimization and prevent the algorithm from learning properly. This is also critical for logistic regression convergence.

## Feature Categories Created

### Performance Features (12 total): HOW GOOD ARE THE TEAMS?

- **Season Stats:** `home_point_diff`, `away_point_diff`, `point_diff_advantage`
- **Matchup Stats:** `home_off_vs_away_def`, `away_off_vs_home_def`, `yards_advantage`
- **Rolling Stats:** `home_rolling_win_pct`, `rolling_win_advantage`, `rolling_point_diff_advantage`

### Team Identity Features (64 total): WHO IS PLAYING?

- **Home Teams:** `home_ARI`, `home_ATL`, `home_BAL`, ..., `home_WAS` (32 features)
- **Away Teams:** `away_ARI`, `away_ATL`, `away_BAL`, ..., `away_WAS` (32 features)

## Data Quality Assurance

### Missing Value Handling
**Functions:** `Imputation`

Early-season games (weeks 1-4) don't have enough prior games for 5-game rolling averages, which creates NaN values that would crash the machine learning algorithm. Imputation with league averages ensures every game has valid features.

### Data Leakage Prevention
**Functions:** `.loc[]`

Can't use future game results to predict past games. Otherwise, the model will look artificially good during training and probably fail in real betting scenarios.

### Chronological Validation
**Functions:** `sort_values()`

Random train/test shuffling would overestimate performance. Since this is time-series data, chronological validation ensures the model can predict future outcomes using only past data, which is exactly what sports betting requires.

## Output Specifications

### Final Dataset Dimensions

- **Samples:** 1,408 NFL games (2020-2024)
- **Features:** 76 total features
  - 12 performance/statistical features
  - 64 team encoding features (32 home + 32 away)
- **Target:** Binary home team victory (0/1)

### Feature Scaling Results

- **Numerical features:** Mean ≈ 0, Standard deviation ≈ 1
- **Binary features:** Values in {0, 1}
- **No missing values:** Complete dataset ready for ML

## Output Files Generated by simplified_nfl_data.py

- `team_season_stats.csv` - Aggregated team performance by season
- `betting_features_X.csv` - ML-ready feature matrix (1,408 × 76)
- `betting_targets_y.csv` - Binary target variable (home team wins)
- `complete_betting_dataset.csv` - Full dataset with all features and targets
- `feature_names.txt` - List of all 76 feature names

## Preprocessing Validation

### Data Integrity Checks

The preprocessing pipeline includes several validation steps to ensure data quality:

- No missing values in final dataset
- All features properly scaled
- Target distribution: 53.8% home wins (realistic)
- Feature correlation analysis completed
- Chronological ordering maintained

### Preprocessing Runtime

- **Execution time:** ~30-60 seconds on standard hardware
- **Memory usage:** <500MB peak



# Model Implementation/Evaluation - Logistic Regression
Our approach consists of four primary phases: data acquisition, preprocessing, feature engineering, and model training. The following section will be discussing the training for our first model: logistic regression, implemented in `nfl_logistic_regression.py`.

Justifications:
- Provides interpretable coefficients
- Outputs probability estimates that are useful for betting thresholds
- Fast to train and thus ideal for initial benchmarking

This model was used as our baseline classifier for predicting whether the home team would win and was trained using **chronologically split** data to avoid data leakage. As referenced in our preprocessing section (split_idx), the first 80% of games were used for training and the final 20% were used for testing. This mimics a real-world betting environment where predictions are made on future games using only past data.

## Logistic Regression Steps

## 1. Data Preparation
**Functions:** `pd.read_csv()`  
Loads CSV files containing NFL game data. A total of 1,408 games with 76 features each (such as team statistics and rolling averages) and the actual results (home team winning or losing) make up the data.

## 2. Model Initialization
**Functions:** `__init__()`  
Establishes the learning rules like how quickly to learn, when to stop, etc (learning rate: 0.1, max iterations: 1,000). 

## 3. Start Training 
**Functions:** `fit()`  
Starts the training procedure. It creates 77 random coefficient guesses (one for each feature plus bias) after first adding a bias column to the data (such as a y-intercept). The starting points are these arbitrary numbers.

## 4. Training Loop (1000 Iterations)
**Functions:** `for loop` inside `fit()`  
Repeats the learning process 1000 times. The model examines every game in each iteration, evaluates how inaccurate its predictions are, and modifies the coefficients to improve accuracy.

### 4a: Calculate Cost Function
**Functions:** `compute_cost()`   
Assesses the model's current performance. uses the current coefficients to make predictions on each of the 1,126 training games and calculates how much the predictions differ from reality.

#### 4a.1: Apply Sigmoid Function
**Functions:** `sigmoid()`  
Creates probabilities between 0 and 1 from the raw mathematical computations.

### 4b: Compute Gradient
**Functions:** `compute_gradient()`  
Determines the direction in which each coefficient should be adjusted. Determine whether and how much each of the 77 coefficients should rise or fall in order to enhance predictions.

### 4c: Weight Update
**Functions:** Direct assignment with `+`, `*` operators  
Modifies the coefficients in accordance with the gradient function's recommendations. Make a tiny adjustment to each coefficient to increase accuracy.

### 4d: Check Convergence
**Functions:** `np.linalg.norm()`  
It checks to see if there is much less change in the coefficients. Stop training if they are stable (convergent). If not, return and use the new coefficients to repeat the loop.

## 5: Store Optimized Coefficients
**Functions:** Assignment to `self.weights`  
This process has produced 76 optimized coefficients after 1000 rounds, such as `away_KC = -0.593`, which collectively predict NFL games with an accuracy of 66.3%. Based on five years of NFL data, these coefficients have "learned" what matters most for winning games.

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


#### Next Steps
- Add **Random Forest** for nonlinear patterns
- Add **Support Vector Machine (SVM)** for the best decision boundaries for game predictions



#

| Name | Midterm Contributions |
|------|----------------------|
| Kevin | README documentation, project introduction, problem statement, and final report writing |
| Thavaisya | Data collection/download, results analysis and visualizations |
| Dishi | Data preprocessing pipeline, feature engineering, and performance evaluation metrics |
| Vivek | Model implementation (logistic regression) |

#
Gantt Chart

![image](https://github.gatech.edu/vbumb3/ml-sports-prediction/blob/main/Gantt_chart.png)

#
References:

[1] "Americans Torn About Sports Betting," Statista, 2021. [Online]. Available: https://www.statista.com/chart/26178/sports-betting-attitudes-us/

[2] T. Angelis et al., "Momentum and betting market efficiency in German soccer," Journal of Sports Economics, vol. 23, no. 4, pp. 456-478, 2022.

[3] "NFL Data via nfl_data_py Package," Official NFL Statistics. [Online]. Available: https://pypi.org/project/nfl-data-py/

[4] Phatak, A. A., Mehta, S., Wieland, F.-G., Jamil, M., Connor, M., Bassek, M., & Memmert, D. (2022). Context is key: Normalization as a novel approach to sport specific preprocessing of KPI’s for Match Analysis in soccer. Scientific Reports, 12(1). https://doi.org/10.1038/s41598-022-05089-y 

[5] Kim, C., Park, J.-H., & Lee, J.-Y. (2024). AI-based betting anomaly detection system to ensure fairness in sports and prevent illegal gambling. Scientific Reports, 14(1). https://doi.org/10.1038/s41598-024-57195-8 

[6] Walsh, C., & Joshi, A. (2024). Machine Learning for Sports Betting: Should Model Selection Be Based on Accuracy or Calibration? https://doi.org/10.2139/ssrn.4705918 

[7] Steyerberg, E. W., Vickers, A. J., Cook, N. R., Gerds, T., Gonen, M., Obuchowski, N., Pencina, M. J., & Kattan, M. W. (2010). Assessing the performance of prediction models. Epidemiology, 21(1), 128–138. https://doi.org/10.1097/ede.0b013e3181c30fb2

[8] R. M. Galekwa et al., "A Systematic Review of Machine Learning in Sports Betting: Techniques, Challenges, and Future Directions," arXiv preprint arXiv:2410.21484, 2024.


