# Business Profit Analysis and Visualization

## Project Overview

This project analyzes business profitability across different geographic locations and over time using a dataset containing revenue and business counts. The analysis focuses on estimating profit metrics, performing statistical testing, regression analysis, and creating informative visualizations.

---

## Features and Workflow

1. **Data Loading and Cleaning**  
   - Loads a CSV dataset with business revenue and count data.  
   - Renames columns for clarity and filters relevant metrics (Total revenue and Total number of businesses).

2. **Data Transformation and Calculation**  
   - Pivots data into a wide format for easier analysis.  
   - Calculates average revenue per business and estimates profit assuming a fixed profit margin (15%).  
   - Applies winsorization to reduce the influence of extreme outliers in estimated profit.  
   - Performs log transformation on estimated profit to normalize data distribution.

3. **Descriptive Statistics and Confidence Intervals**  
   - Computes mean, median, standard deviation, and count of log-transformed estimated profit by location type.  
   - Calculates 95% confidence intervals for these statistics.

4. **Statistical Testing**  
   - Performs independent t-tests comparing log-estimated profits between different location types (e.g., urban vs rural).

5. **Regression Analysis**  
   - Builds an Ordinary Least Squares (OLS) regression model to assess the effect of location type on log-estimated profit.

6. **Visualization**  
   - Generates histograms showing the distribution of log-estimated profits by location.  
   - Creates connected scatter plots illustrating trends in log-estimated profit over years, including percentage differences between locations.

---

## Technologies and Libraries

- Python 3.x  
- Pandas for data manipulation  
- NumPy for numerical operations  
- Seaborn and Matplotlib for visualization  
- SciPy for statistical testing  
- Statsmodels for regression analysis  

---

## Usage

1. Place your dataset file named `Dataset.csv` in the project directory.  
2. Run the Python script to perform analysis and generate outputs.  
3. Outputs include CSV and TXT files summarizing statistics, test results, regression model, and PNG files with visualizations.

---

## Output Files

- `Log_Profit_Stats.csv` and `.txt`: Summary statistics for log-estimated profit.  
- `Log_Profit_CI.csv` and `.txt`: Confidence intervals for log-estimated profit.  
- `T_test_log_profit.txt`: Results of t-tests comparing location types.  
- `Regression_Log_Profit.txt`: Summary of regression model results.  
- `Log_Estimated_Profit_Distribution.png`: Histogram of log-estimated profits by location.  
- `Connected_Scatter_Log_Profit.png`: Connected scatter plot of yearly log-estimated profits.

---

## Project Insights

This analysis provides a robust approach to estimating and comparing business profitability in urban versus rural or small-town locations over time. The combination of statistical testing, regression modeling, and visualization allows for deeper understanding of spatial and temporal trends in profitability.

---

## License

This project is open source and available under the MIT License.

---

## Contact

For questions or feedback, please contact Aisha.

