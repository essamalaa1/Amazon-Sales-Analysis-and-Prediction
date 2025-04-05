# Amazon Sale Report Analysis and Sales Dashboard
This project analyzes an Amazon sales report dataset, performing data cleaning, exploratory analysis, predictive modeling, and dashboard development. The code is structured into several key steps as described below:

## 1. Exploratory Data Analysis (EDA)
### Data Loading:
The dataset is imported using pandas.read_csv and initial inspections (shape, head, datatypes) are performed.

### Duplicate & Null Check:
The code checks for duplicate rows and counts missing values, helping to identify columns with single unique values or high missingness.

### Initial Conclusions from EDA:
Based on the EDA, the following actions were determined:

Drop columns with only one unique value (e.g., currency, ship-country, etc.).

Scale numerical columns (Qty, Amount) due to differences in distribution.

Convert the Date column to datetime and extract Month and Day.

Drop or impute columns with high missing values (e.g., fulfilled-by and promotion-ids).

## 2. Data Preprocessing
### Cleaning the Dataset:

Dropping irrelevant or redundant columns (e.g., Order ID, Unnamed: 22).

Converting the Date field into datetime format and extracting useful features (Year, Month, Day), then dropping the Year and original Date columns.

### Handling Missing Values:

Numeric Imputation:
The Amount column is imputed using the KNN imputer (KNNImputer) from scikit-learn, ensuring that the continuous data is filled based on nearest neighbors.

Categorical Imputation:
For the Courier Status column, missing values are imputed probabilistically based on the existing class distribution.

Dropping Nulls:
Rows with missing ship-postal-code values are dropped.

### Feature Encoding and Scaling:

Label Encoding:
Categorical and Boolean columns are encoded using LabelEncoder.

Feature Scaling:
The numerical features (excluding the target Courier Status) are scaled using StandardScaler to normalize the data.

### Outlier Detection and Removal:

Outliers in the Amount and Qty columns are identified using the interquartile range (IQR) method and visualized using box plots.

A KMeans clustering approach is used to determine the optimal number of clusters (using the Elbow Method) and to remove outliers based on distance thresholds.

## 3. Data Visualization
### Exploratory Visualizations:

Distribution plots (histograms, bar plots) are generated for various features including order status, size, courier status, and sales trends over time.

A correlation matrix heatmap is produced to identify interrelationships among features.

### Insights Visualization:

Additional plots provide insights into regional sales, fulfillment types, and shipping service levels.

## 4. Predictive Modeling
### Addressing Class Imbalance:

The target feature Courier Status is highly imbalanced. To address this, the SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the classes.

### Modeling Process:

Model Selection:
A Logistic Regression model is chosen for classification.

Feature Selection:
Forward feature selection using the Sequential Feature Selector (SFS) from mlxtend is applied to identify the most important features.

Training and Evaluation:

The data is split into training and test sets.

The Logistic Regression model is trained and predictions are made on both sets.

Performance is evaluated using confusion matrices, precision, recall, F1-score, and log loss.

Cross-validation (StratifiedKFold with 5 splits) confirms model stability and generalization with minimal overfitting.

## 5. Dashboard Development
### Interactive Dashboard with Dash:

A dashboard is built using the Dash framework along with Plotly and Dash Bootstrap Components.

Dashboard Features:

A main line graph displays daily order trends for each month.

Dropdown controls allow users to filter and view state and city sales based on the selected month.

Bar charts show the top 10 states and cities by sales amount.

### Customization:
The dashboard design is customized with predefined colors and layouts to ensure an appealing user interface.

## Technologies and Libraries Used
## Data Handling and Analysis:

pandas, numpy

## Visualization:

seaborn, matplotlib, plotly

## Preprocessing & Imputation:

sklearn.preprocessing, sklearn.impute.KNNImputer

## Imbalanced Data Handling:

imblearn.over_sampling.SMOTE

## Modeling:

sklearn.linear_model.LogisticRegression

mlxtend.feature_selection.SequentialFeatureSelector

## Clustering & Outlier Detection:

sklearn.cluster.KMeans

## Dashboard Development:

dash, dash_bootstrap_components
