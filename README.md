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
| Logistic regression | 0.54 | 0.39 | 0.78 | 0.52 |
| Decision tree | 0.68 | 0.49 | 0.24 | 0.32 |

The models were evaluated using a chronological 80/20 split: the first 80% of the dataset's rows were used for training and the final 20% for testing. Because the dataset has no year column, the split relies on the original row order and cannot identify an exact future period.

The logistic regression found 78% of customers who actually subscribed, but 39% of its `yes` predictions were correct. It is therefore useful when the goal is to identify as many potential subscribers as possible.

The decision tree had higher accuracy and precision, but found only 24% of actual subscribers. It is useful when reducing unnecessary calls is more important.

Precision and recall are more informative than accuracy alone for this imbalanced target: most customers in the dataset belong to the `no` class.

## Notes and limitations

- The dataset includes categorical values such as `unknown`; these are treated as categories, not as pandas missing values.
- In `bank-full.csv`, `pdays = -1` indicates that a customer was not previously contacted, rather than a missing value.
- The chronological 80/20 split is an initial approximation because no year column is available.
- The results do not guarantee performance on future campaigns.
- The analysis identifies patterns in the data; it does not establish that any feature causes a customer to subscribe.

## Predicting future customers

A trained model can predict a new customer only when the input has the same structure used during training.

- Use the 15 feature columns used for training: `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `campaign`, `pdays`, `previous`, and `poutcome`.
- Do not provide `y` to the model. `y` is the result that the model is trying to predict.
- Do not provide `duration`. It is only known after a call and would make the model unsuitable for pre-call prediction.
- Keep column names, data types and categorical values consistent with the training dataset.
- Save the preprocessing and trained model together in a pipeline. When using `joblib`, the saved pipeline can handle the preprocessing and prediction for future customers.

```python
# Example after creating and training the model:
import joblib

joblib.dump(model, "bank_marketing_model.joblib")

# Example for new customers:
loaded_model = joblib.load("bank_marketing_model.joblib")
prediction = loaded_model.predict(new_customers)
probability = loaded_model.predict_proba(new_customers)
```

Mock data can be used to test the prediction code, but real campaign data and the actual outcome (`y`) are required to measure future performance.

## Run the notebook

1. Download the **Bank Marketing** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing).
2. Place `bank-full.csv` in the project directory.
3. Install Python, then install the project dependencies with `pip install -r requirements.txt`.
4. Open and run `Classification-Bank_full.csv.ipynb` from top to bottom.


## Dataset citation

Moro, S., Laureano, R., & Cortez, P. (2011). *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*. Proceedings of the European Simulation and Modelling Conference (ESM'2011), pp. 117–121. https://hdl.handle.net/1822/14838
