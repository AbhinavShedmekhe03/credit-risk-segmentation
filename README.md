
# Credit Risk Prediction and Customer Segmentation using Logistic Regression and KMeans Clustering

## 📅 Project Snapshot
The goal of this project was to dig into real-world financial data and figure out two things: 
1. Can we predict whether someone is a good or bad credit risk?
2. Can we group similar customer profiles even when we don’t know their credit history?

To answer both, we used two machine learning approaches:
- **Logistic Regression** for prediction (supervised learning)
- **KMeans Clustering** for segmentation (unsupervised learning)

---

## 📖 Dataset at a Glance
- **Source**: UCI Statlog German Credit Data
- **File Used**: `german.data`
- **Total Records**: 1000 customers
- **Original Features**: 20
- **Target Column**: `CreditRisk` (1 = Good credit, 2 = Bad credit)

Out of the 20 features:
- 13 were **categorical**, labeled as codes like `A11`, `A34` etc.
- 7 were **numerical**, including variables like age, credit amount, and duration.

We referred to the `german.doc` file to decode what each categorical code meant and assigned clean column names for clarity.

---

## 🧐 What We Set Out To Do
Real-world credit risk problems require:
- Predicting if a person is likely to default (Logistic Regression)
- Finding hidden patterns or groups without needing labels (KMeans Clustering)

This dual-track project allowed us to tackle both problems using the same dataset.

---

## ⚖️ How We Prepped the Data
1. **Loaded** the raw data and mapped it with readable column names.
2. Performed initial checks – no missing values found.
3. Ran **EDA** to get a feel for the data:
   - Found a 70/30 split between good and bad credit cases.
   - Analyzed how categories relate to risk using cross-tabs.
4. Split features into **categorical and numerical** groups.
5. Applied **one-hot encoding** to convert categorical variables into numeric format.
6. Used `drop_first=True` in encoding to avoid multicollinearity.
7. Final dataset expanded to 49 columns post-encoding.
8. Applied **StandardScaler** to normalize all features (important for both Logistic Regression and KMeans).

---

## 📊 Model 1: Logistic Regression
This was our prediction engine – used to predict if a customer is high or low risk.

- **Why we chose it**: Straightforward, interpretable, and commonly used in finance.
- Trained on 80% of the data (stratified split)
- Used accuracy, confusion matrix, and classification report to evaluate
- Final accuracy: **~78%**, which is solid for a simple model

> Logistic Regression helped us build a practical model that could generalize well to new data.

---

## 🔄 Model 2: KMeans Clustering
This helped us find **natural customer segments** without using credit labels.

- **Why KMeans?**
  - Doesn't require target labels
  - Works well with scaled, numeric data

**Steps followed:**
1. Used **Elbow Method** to choose `k=2` (sweet spot before returns diminished)
2. Trained KMeans on the full feature set
3. Reduced dimensionality using **PCA** for 2D visualization (PC1 and PC2 axes)
4. Compared cluster labels with actual credit labels

**What we found:**
- Cluster 0: 41.67% bad credit
- Cluster 1: 29.85% bad credit
- Without knowing anything about credit labels, KMeans was able to separate high and low risk segments reasonably well.

---

## 🌐 Business Relevance
| Goal | Technique | Impact |
|------|-----------|--------|
| Predict credit defaults | Logistic Regression | Helps take proactive decisions on loan approval |
| Segment customers | KMeans Clustering | Helps understand customer behavior without needing credit labels |

---

## 📄 Final Thoughts
This project was a hands-on walk through the entire machine learning lifecycle:
- Cleaned and prepared a real-world credit dataset
- Tackled a classification problem using logistic regression
- Explored patterns using clustering and PCA
- Compared unsupervised insights with actual labels

Together, these models give us a clear way to predict and understand credit behavior – super relevant for roles in **finance, analytics, and risk modeling**.

---

## 🚀 Next Steps
- Add ROC curve and confusion matrix heatmap
- Deploy results via dashboard (e.g., Streamlit)
- Push the notebook + summary to GitHub and LinkedIn
