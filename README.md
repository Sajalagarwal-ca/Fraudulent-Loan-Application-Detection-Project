# Project Report: Fraudulent Loan Application Detection

## 1. Problem Definition

The goal of this project is to detect fraudulent loan applications. In financial institutions, fraud detection is critical as it helps in identifying illegitimate loan applications that could result in financial losses. The challenge is to develop a machine learning model that can accurately classify applications as either fraudulent or legitimate based on various factors such as credit history, income, and loan type.

### Key Objective:
To build a classification model with high performance (F1-score above 0.75) for detecting fraudulent applications.

---

## 2. Data Understanding

The dataset used for this project contains various features that describe loan applications. Key features include:

- `credit.policy`: Whether the applicant meets the credit underwriting criteria.
- `purpose`: The stated purpose of the loan (e.g., debt consolidation, credit card).
- `int.rate`: Interest rate of the loan.
- `installment`: Monthly installment of the loan.
- `log.annual.inc`: Log of the annual income.
- `dti`: Debt-to-income ratio.
- `fico`: FICO credit score.
- `revol.bal`: Revolving balance.
- `inq.last.6mths`: Number of inquiries in the last 6 months.
- `not.fully.paid`: Target variable indicating whether the loan was fully repaid.

---

## 3. Approach

### a. Data Preprocessing
- The data was loaded and analyzed for missing values and inconsistencies. Basic exploratory data analysis (EDA) was conducted using libraries like `pandas`, `matplotlib`, and `seaborn`.
- Oversampling was applied using `SMOTE` to handle class imbalance between fraudulent and legitimate applications.

### b. Modeling
Several machine learning algorithms were considered:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost, CatBoost, and LightGBM

Each model was evaluated using metrics like accuracy, F1-score, confusion matrix, and classification report.

### c. Cross-Validation and Hyperparameter Tuning
- `GridSearchCV` was used for tuning hyperparameters to optimize model performance.

### d. Feature Engineering
- Various transformations and scaling techniques were applied to the dataset, especially for continuous variables like `int.rate` and `installment`.

---

## 4. Results

The Random Forest model performed the best with an F1-score of above 0.75. The following metrics were considered for evaluating the models:

- **Accuracy**: Proportion of correct predictions.
- **F1-Score**: Harmonic mean of precision and recall, a key metric for imbalanced datasets.
- **Confusion Matrix**: Breakdown of true positives, true negatives, false positives, and false negatives.

### Key Findings:
- Credit score (`fico`), `dti`, and loan purpose were strong predictors of whether a loan application was fraudulent.
- Imbalanced classes were handled using `SMOTE`, which improved model performance significantly.

---

## 5. Future Improvements

- **Feature Engineering**: More sophisticated feature engineering techniques could be applied, such as creating interaction terms between variables like `fico` and `int.rate`.
- **Anomaly Detection**: Implementing unsupervised anomaly detection methods might help to flag unusual patterns in applications that are not captured by supervised learning models.
- **Model Ensemble**: Combining the strengths of multiple models via stacking or boosting could improve performance further.
- **Real-time Detection**: Implementing this solution in a real-time detection environment to catch fraudulent applications as they are submitted.

---

## 6. Assumptions and Challenges

- **Assumptions**: It was assumed that the dataset accurately represents the types of loan applications typically seen by financial institutions.
- **Challenges**:
  - **Class Imbalance**: The dataset had far fewer fraudulent applications compared to legitimate ones, making it difficult for standard classifiers to perform well. This was addressed using the SMOTE technique.
  - **Data Quality**: Some features required cleaning, and dealing with missing values presented a challenge.

---