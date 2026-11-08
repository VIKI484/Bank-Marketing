# Bank Marketing: Predicting Term Deposit Subscriptions

## Project overview

This project uses machine learning to predict whether a bank customer subscribed to a term deposit after a direct marketing campaign. The data comes from phone-based campaigns conducted by a Portuguese banking institution.

The analysis uses **`bank-full.csv`**, which contains 45,211 observations, 16 input features, and the target variable `y`. The target has two classes: `yes` and `no`.

## Workflow

- Inspected the dataset's size, columns, and data types.
- Checked for missing values, duplicate rows, and unusual numeric values.
- Examined the distribution of the target classes.
- Prepared numeric and categorical features for modeling.
- Compared logistic regression with a decision tree.
- Evaluated both models using precision, recall, and F1 score for the `yes` class, as well as accuracy.

The `duration` feature (call duration) was excluded because it is not known before a call takes place. Including it would make the model less representative of a real prediction made before contacting a customer.

## Results

| Model | Accuracy | Precision (`yes`) | Recall (`yes`) | F1 (`yes`) |
|---|---:|---:|---:|---:|
| Logistic regression | 0.75 | 0.27 | 0.62 | 0.37 |
| Decision tree | 0.80 | 0.30 | 0.56 | 0.39 |

The decision tree achieved a slightly higher F1 score, while logistic regression identified a larger proportion of customers who actually subscribed (higher recall). Precision was low for both models, meaning many customers predicted as `yes` did not subscribe.

Accuracy alone can be misleading because the target classes are imbalanced: most customers in this dataset are in the `no` class. A model can achieve high accuracy by mostly predicting `no` while still failing to identify many `yes` cases. The choice between models therefore depends on the goal: finding more potential subscribers favors recall, while reducing false positive predictions favors precision.

## Notes and limitations

- The dataset includes categorical values such as `unknown`; these are treated as categories, not as pandas missing values.
- In `bank-full.csv`, `pdays = -1` indicates that a customer was not previously contacted, rather than a missing value.
- The results are based on one train/test split and should be treated as an initial comparison, not a guarantee of performance on future campaigns.
- The analysis identifies patterns in the data; it does not establish that any feature causes a customer to subscribe.

## Run the notebook

1. Download the **Bank Marketing** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing).
2. Place `bank-full.csv` in the project directory.
3. Install Python, Jupyter, pandas, and scikit-learn.
4. Open and run `Classification-Bank_full.csv.ipynb` from top to bottom.

## Dataset citation

Moro, S., Laureano, R., & Cortez, P. (2011). *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*. Proceedings of the European Simulation and Modelling Conference (ESM'2011), pp. 117–121. https://hdl.handle.net/1822/14838
