# Proposal


From the National Football League to the Ultimate Fighting Championship, sports betting has become increasingly popular amongst American youth. One survey by Statista showed that 30% of respondents had put money on sports with the NFL being most popular[1].


Sports betting can be seen as a predatory industry, as it is addictive and almost all people lose money. The spreads and betting odds are largely manipulated by social media trends and analysis of how much money is being placed on each team. A study on German soccer showed [2] that significant portions of betting odds were based on perceived team momentum, which has no correlation with actual success. This ensures companies like DraftKings and FanDuel can make the largest profit possible. Therefore, constructing a model using only objective metrics like player stats should be more accurate and enable consumers to make a profit.


Our analysis utilizes NFL team offensive and defensive statistics spanning from 2020 to 2024 [3]. The dataset includes season-end averages for key metrics such as points scored, total yards, plays run, yards per play, turnovers, first downs, and comprehensive passing statistics including completions, attempts, passing yards, touchdowns, and interceptions. This 5-year historical dataset provides extensive training data across recent seasons, ensuring relevance to modern NFL gameplay and strategy.


To prepare data, missing numeric values will be imputed using median to account for outliers. Duplicates will be removed to ensure each entry is unique. We will engineer features including 5-game rolling averages of point differentials and yards/play, plus relative performance metrics capturing team strengths and weaknesses relative to opponents. One-hot coding will convert team names into numerical values for algorithm compatibility. Since algorithms like logistic regression use gradient-based optimization, features with huge scales cause erratic training. Therefore, continuous features will be standardized [(original_value - mean) / SD] to give each feature equal importance [4].


The main supervised learning algorithms we are implementing are logistic regression, random forest, and support vector machines. Logistic regression serves as baseline, fitting weighted sums of input features to estimate win probabilities. Coefficient weights establish how each statistic drives game outcomes, especially helpful during initial modeling stages. Random forest determines nonlinear interactions by averaging votes of many decision trees, providing robust probability estimates and built-in feature ranking. SVM finds the optimal decision boundary separating wins from losses, predicting which side new games will fall under [5].


We will evaluate our model using three quantitative metrics. First, prediction accuracy will be determined by calculating the percentage of correct game outcome predictions across test games against sportsbook odds and spreads. Then, Return on Investment (ROI) will measure profitability by simulating bets placed when our model's prediction probability exceeds implied probability from sportsbook odds [6]. The final metric will be Brier score, which evaluates probabilistic prediction quality by calculating how close predicted probabilities are to actual outcomes [7].


Our primary goal is achieving 55-60% prediction accuracy based on existing NFL prediction research [8], while maintaining positive ROI over multiple seasons. This target exceeds the 52.4% threshold needed to overcome typical -110 betting odds. From an ethical standpoint, we aim to create a statistic-driven alternative to perception-manipulated betting lines. Regarding sustainability, our model emphasizes statistical validity over short-term gains, encouraging disciplined approaches to sports betting.

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

#

| Name | Proposal Contributions |
|------|----------------------|
| Eshaan | Intro, background, problem |
| Thavaisya | Methods (ML algorithms) |
| Dishi | Methods (Preprocessing) and quantitative metrics |
| Vivek | Dataset info, Potential results and discussion |

#
Gantt Chart

![image](https://github.gatech.edu/vbumb3/ml-sports-prediction/blob/main/Gantt_chart.png)


