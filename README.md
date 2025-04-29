# Customer-Churn-Prediction
This Project aims to predict the customer churn. We have used the Random forest and XGBoost as a hybrid model, finally estimated using the Logistic Regression. 

```markdown
# 🚀 Customer Churn Prediction

A complete end-to-end solution to predict which customers will cancel a subscription-based service. This repo includes:

1. **Data preprocessing**  
2. **Hybrid stacking model** (Random Forest + XGBoost base learners, Logistic Regression meta-learner)  
3. **Artifact saving/loading**  
4. **Single & batch prediction scripts**  
5. **Streamlit web app** for interactive use  

---

## 📂 Repository Structure

```
├── data/
│   ├── Churn_Modelling.csv       ← Raw dataset for training
│   └── new/                      ← Example CSVs for batch testing in Streamlit
│
├── Model files/
│   ├── churn_model.pkl           ← Trained stacking model
│   ├── scaler.pkl                ← StandardScaler object
│   └── label_encoders.pkl        ← LabelEncoder mapping for Geography & Gender
│
├── task3.ipynb                   ← Jupyter notebook with EDA & model training
├── train_and_save.py             ← Standalone training & artifact-saving script
├── predict_churn_single.py       ← CLI script for one-customer prediction
├── predict_churn_batch.py        ← CLI script for batch CSV prediction
└── task3_streamlit.py            ← Streamlit app for web-based predictions
```


## 📖 Theoretical Overview

### 1. Data Preprocessing

- **Dropping identifiers**  
  We remove `RowNumber`, `CustomerId`, and `Surname` before training because they carry no predictive power.
- **Imputation**  
  Missing numeric values are filled with the **median** of each column. This is robust to outliers.
- **Encoding**  
  Categorical features (`Geography`, `Gender`) are converted to integers via `LabelEncoder`.  
- **Scaling**  
  Continuous features are standardized (zero mean, unit variance) using `StandardScaler`. This prevents features with large ranges from dominating the model.

### 2. Hybrid Stacking Model

1. **Base Learners**  
   - **Random Forest**: an ensemble of decision trees that reduces variance by averaging.  
   - **XGBoost**: a gradient-boosted tree model that sequentially corrects errors, great for tabular data.  
2. **Meta-Learner**  
   - **Logistic Regression**: takes the base learners’ predictions as input and learns an optimal combination.  
3. **Stacking Workflow**  
   - Perform 5-fold cross-validation on base learners to generate out-of-fold predictions.  
   - Train the logistic regression on these predictions to produce the final output.  

This approach often outperforms any single model by leveraging their complementary strengths.


## 🛠️ Code Explanation

### `train_and_save.py`

1. **Load data**  
   ```python
   df = pd.read_csv('data/Churn_Modelling.csv')
   ```
2. **Preprocess**  
   - Drop unused columns  
   - Impute numerics:  
     ```python
     df[num_cols] = SimpleImputer('median').fit_transform(df[num_cols])
     ```  
   - Encode categoricals:  
     ```python
     le_geo = LabelEncoder().fit(df['Geography'])
     df['Geography'] = le_geo.transform(df['Geography'])
     ```
3. **Split**  
   ```python
   X_train, X_test, y_train, y_test = train_test_split(..., stratify=y)
   ```
4. **Scale**  
   ```python
   scaler = StandardScaler().fit(X_train)
   X_train = scaler.transform(X_train)
   ```
5. **Build & train stacking model**  
   ```python
   base_learners = [
       ('rf', RandomForestClassifier(...)),
       ('xgb', XGBClassifier(...))
   ]
   stack_model = StackingClassifier(
       estimators=base_learners,
       final_estimator=LogisticRegression(),
       cv=5
   ).fit(X_train, y_train)
   ```
6. **Evaluate**  
   Compute accuracy, ROC-AUC, classification report, and confusion matrix on train/test.
7. **Save artifacts**  
   ```python
   joblib.dump(stack_model, 'Model files/churn_model.pkl')
   joblib.dump(scaler,      'Model files/scaler.pkl')
   joblib.dump({'Geography': le_geo, 'Gender': le_gender}, 'Model files/label_encoders.pkl')
   ```

### `predict_churn_single.py`

- **Loads** the saved model, scaler, and encoders
- **Prompts** user for each feature via `input()`  
- **Encodes**, **scales**, and **predicts**  
- **Prints** “Will churn” / “Will not churn” plus probability

### `predict_churn_batch.py`

- **Loads** artifacts
- **Reads** a CSV of new customers
- **Preprocesses** (drop identifiers, impute, encode, scale)
- **Predicts** on all rows
- **Writes** back a CSV with columns:
  ```
  CustomerId, Surname, ChurnPrediction, ChurnProbability
  ```

### `task3_streamlit.py`

- **Two modes** via radio button:  
  - **Single Prediction**: interactive form → immediate result  
  - **Batch Prediction**: CSV uploader → DataFrame display  
- **Clears** the other output on mode switch for a clean interface


## 🚀 Running the App

1. Install dependencies:  
   ```bash
   pip install pandas numpy scikit-learn xgboost joblib streamlit
   ```
2. **Train & save** artifacts (if you haven’t):  
   ```bash
   python train_and_save.py
   ```
3. **Run Streamlit**:  
   ```bash
   streamlit run task3_streamlit.py
   ```
4. **Use CLI scripts** for quick local tests:  
   ```bash
   python predict_churn_single.py
   python predict_churn_batch.py --input data/new/sample.csv
   ```


## 💾 Download Model Artifacts

- [churn_model.pkl](Model%20files/churn_model.pkl)  
- [scaler.pkl](Model%20files/scaler.pkl)  
- [label_encoders.pkl](Model%20files/label_encoders.pkl)  

Feel free to ⭐ the repo and open issues if you have questions or ideas!
