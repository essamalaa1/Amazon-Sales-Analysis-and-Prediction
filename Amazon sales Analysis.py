import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, log_loss, accuracy_score
from sklearn.impute import KNNImputer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.cluster import KMeans

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import plotly.graph_objs as go
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
### Step 1 :  Exploratory Data Analysis (EDA)
data = pd.read_csv("Amazon Sale Report.csv",low_memory=False)
data.shape
data.head()
data.dtypes
duplicates = data.duplicated()

num_duplicates = duplicates.sum()
print(f'Number of duplicate rows: {num_duplicates}')
numeric_summary = data.describe()
numeric_summary
categorical_summary = data.describe(include=[object,bool])
categorical_summary
print(data.isnull().sum().sum())
null_counts = data.isnull().sum()
columns_with_null = null_counts[null_counts > 0]
columns_with_null
# ```c++
# From the above EDA we can conclude :
# 1)columns [Currency , ship-country , Fulfilled by , Unnamed-32] contains only one unique value , so it can be dropped 
# 2)columns [Qty , Amount] need to be scaled due to huge difference in their distribution 
# 3)column date need to be converted from object data type to datetime 
# 4)column postal code is numeric column but its characteristic is for categorical column 
# 5)columns [ship-country , ship-city , ship-state , ship-postal-code] are considered as hierarchical features starting from country to postal-code , so we can keep the column with the lowest granularity which is postal code and drop the others   
# 6)column 'fulfilled-by' has 89698 null values which means that approximately 69.5% of the column is missing , so this column will be dropped as imputing it will be misleading , and also it contains only one variable 
# 7)columns 'promotion-ids' has null values 49153  which will be dropped also due to large number of missing values**

# ```
### Step 2 : Data Preprocessing
cleaned_data = data.drop(columns=['currency','ship-country','fulfilled-by','Unnamed: 22','ship-city','ship-state','promotion-ids'])
cleaned_data = cleaned_data.drop(['index', 'Order ID'],axis=1)
cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'], format='%m-%d-%y')
cleaned_data['Year'] = cleaned_data['Date'].dt.year
cleaned_data['Month'] = cleaned_data['Date'].dt.month
cleaned_data['Day'] = cleaned_data['Date'].dt.day
print(cleaned_data['Year'].nunique())
print(cleaned_data['Month'].nunique())
print(cleaned_data['Day'].nunique())
# Year column will be dropped since it has only one value and the main date column will be dropped as we replaced it with 2 columns one for mounth and other for day
cleaned_data = cleaned_data.drop(['Year','Date'],axis=1)
cleaned_data
cleaned_data.dtypes
cleaned_data
print("count of all nulls : ")
print(cleaned_data.isnull().sum().sum())
print("")
null_counts = cleaned_data.isnull().sum()
columns_with_null = null_counts[null_counts > 0]
print("Columns with nulls : ")
print(columns_with_null)
# distribution of Amount before imputation

mean_Amount = data['Amount'].mean()
std_Amount = data['Amount'].std()

sns.histplot(cleaned_data['Amount'], kde=True)

plt.title('Distribution of Amount before imputation')
plt.xlabel('Amount')
plt.ylabel('Frequency')

plt.axvline(mean_Amount, color='r', linestyle='--', label=f'Mean: {mean_Amount:.2f}')
plt.axvline(mean_Amount + std_Amount, color='g', linestyle='--', label=f'Std Dev: {std_Amount:.2f}')
plt.axvline(mean_Amount - std_Amount, color='g', linestyle='--')

plt.legend()

plt.show()
# Imputation Using KNN
imputer = KNNImputer(n_neighbors=5)
cleaned_data['Amount'] = imputer.fit_transform(cleaned_data[['Amount']])
# distribution of Amount after imputation

mean_Amount = cleaned_data['Amount'].mean()
std_Amount = cleaned_data['Amount'].std()

sns.histplot(cleaned_data['Amount'], kde=True)

plt.title('Distribution of Amount after imputation')
plt.xlabel('Amount')
plt.ylabel('Frequency')

plt.axvline(mean_Amount, color='r', linestyle='--', label=f'Mean: {mean_Amount:.2f}')
plt.axvline(mean_Amount + std_Amount, color='g', linestyle='--', label=f'Std Dev: {std_Amount:.2f}')
plt.axvline(mean_Amount - std_Amount, color='g', linestyle='--')

plt.legend()

plt.show()
print(cleaned_data.isnull().sum().sum())
null_counts = cleaned_data.isnull().sum()
columns_with_null = null_counts[null_counts > 0]
columns_with_null
# Distribution of Courier Status before imputation
Courier_Status_count = cleaned_data['Courier Status'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=Courier_Status_count.index, y=Courier_Status_count.values, palette='viridis')

plt.title('Distribution of Courier_Status before imputation')
plt.xlabel('Courier_Status_count')
plt.ylabel('Count')

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
Courier_Status_count
# Imputation of Courier Status using the probabillity of each class
Shipped = 109487
Unshipped = 6681
Cancelled = 5935

total_non_missing = Shipped + Unshipped + Cancelled

prop_Shipped = Shipped / total_non_missing
prop_Unshipped = Unshipped / total_non_missing
prop_Cancelled = Cancelled / total_non_missing


def impute_category(row):
    if pd.isna(row['Courier Status']):
        rand_val = np.random.random()
        if rand_val < prop_Shipped:
            return 'Shipped'
        elif rand_val < prop_Shipped + prop_Unshipped:
            return 'Unshipped'
        else:
            return 'Cancelled'
    else:
        return row['Courier Status']

cleaned_data['Courier Status'] = cleaned_data.apply(impute_category, axis=1)
# Distribution of Courier Status after imputation
Courier_Status_count = cleaned_data['Courier Status'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=Courier_Status_count.index, y=Courier_Status_count.values, palette='viridis')

plt.title('Distribution of Courier_Status aftr imputation')
plt.xlabel('Courier_Status_count')
plt.ylabel('Count')

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
Courier_Status_count
print(cleaned_data.isnull().sum().sum())
null_counts = cleaned_data.isnull().sum()
columns_with_null = null_counts[null_counts > 0]
columns_with_null
cleaned_data = cleaned_data.dropna(subset=['ship-postal-code'])
cleaned_data
print(cleaned_data.isnull().sum().sum())
null_counts = cleaned_data.isnull().sum()
columns_with_null = null_counts[null_counts > 0]
columns_with_null
cleaned_data.dtypes
label_encoder = LabelEncoder()

for column in cleaned_data.select_dtypes(include=['object','bool']).columns:
    cleaned_data[column] = label_encoder.fit_transform(cleaned_data[column])

cleaned_data.head()
cleaned_data.describe()
scaler = StandardScaler()

# excluding 'Courier Status' from scaling as it is our target column

numerical_features = cleaned_data.drop(['Courier Status'],axis=1).select_dtypes(include=['float64', 'int64', 'int32', 'float32']).columns
cleaned_data.loc[:, numerical_features] = scaler.fit_transform(cleaned_data.loc[:, numerical_features])

cleaned_data
def count_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_count = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
    return outlier_count

outlier_counts = {}
outlier_counts['Amount'] = count_outliers(cleaned_data, 'Amount')
outlier_counts['Qty'] = count_outliers(cleaned_data, 'Qty')

print("Outlier counts:")
for column, count in outlier_counts.items():
    print(f"{column}: {count}")

sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.boxplot(data=cleaned_data, y='Qty', ax=axes[0])
axes[0].set_title('Box Plot for Qty')

sns.boxplot(data=cleaned_data, y='Amount', ax=axes[1])
axes[1].set_title('Box Plot for Amount')

plt.tight_layout()
plt.show()
# X = cleaned_data.drop(columns=['Courier Status'])
# y = cleaned_data['Courier Status']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model = LogisticRegression(max_iter=1000, random_state=42)

# sfs = SFS(model,
#           k_features=(1, len(X.columns)),
#           forward=True,
#           scoring='accuracy',
#           cv=5)

# sfs.fit(X_train, y_train)

# print("Selected features:")
# print(sfs.k_feature_names_)
# print("Accuracy of the selected feature subset:", sfs.k_score_)

# X_train_selected = sfs.transform(X_train)
# X_test_selected = sfs.transform(X_test)

# model.fit(X_train_selected, y_train)

# y_pred = model.predict(X_test_selected)

# accuracy = accuracy_score(y_test, y_pred)
# print("Final accuracy on test set:", accuracy)
# ```c++
# the above cell takes 8 minutes runtime but the ouput of it is 
# Selected features:('Status', 'Fulfilment', 'Sales Channel ', 'Style', 'SKU', 'Category', 'ASIN', 'Qty', 'Amount', 'ship-postal-code', 'B2B', 'Month', 'Day')
# Accuracy of the selected feature subset: 0.9920894326257302
# Final accuracy on test set: 0.9915208231005869
# ```
cleaned_data = cleaned_data[['Status', 'Fulfilment', 'Sales Channel ', 'Style', 'SKU', 'Category', 'ASIN', 'Qty', 'Amount', 'ship-postal-code', 'B2B', 'Month', 'Day','Courier Status']]
cleaned_data
# feature selected based on forward feature selection
corr_matrix = cleaned_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
# Sku and style are nearly identical so one of them will be dropped (we will choose SKU)
cleaned_data = cleaned_data.drop(columns=['SKU'])
# from sklearn.neighbors import NearestNeighbors

# def hopkins(X):
#     d = X.shape[1]
#     n = len(X)
#     m = int(n)
#     nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
#     rand_X = np.random.random((m, d))
#     u_distances = []
#     for rand_x in rand_X:
#         u_distances.append(nbrs.kneighbors([rand_x], 2, return_distance=True)[0][0][1])
#     w_distances = []
#     sample_indices = np.random.choice(np.arange(n), m, replace=False)
#     for sample_index in sample_indices:
#         w_distances.append(nbrs.kneighbors([X.iloc[sample_index]], 2, return_distance=True)[0][0][1])
#     H = np.sum(u_distances) / (np.sum(u_distances) + np.sum(w_distances))
#     return H

# hopkins_statistic = hopkins(cleaned_data)
# print(f"Hopkins statistic: {hopkins_statistic}")
# This Cell Takes 3 minutes when we run it on the whole dataset ,
# The Hopkins statistic is 0.891619922810035 which skows that the data has a good tend to be clustered
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(cleaned_data)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10,6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of clusters, k')
plt.ylabel('Within-cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()
# The elbow method shows that k = 6
k = 6
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(cleaned_data)
distances = kmeans.transform(cleaned_data)
nearest_distances = np.min(distances, axis=1)
threshold = np.mean(nearest_distances) + 2 * np.std(nearest_distances)
outliers_indices = np.where(nearest_distances > threshold)[0]
outliers_mask = np.zeros(len(cleaned_data), dtype=bool)
outliers_mask[outliers_indices] = True
print("Shape of outliers:", outliers_indices.shape)
cleaned_data = cleaned_data[~outliers_mask]
print("Cleaned data without outliers:")
cleaned_data
# Outliers are dropped
### Step 3 Data Visuilization
df = pd.read_csv("Amazon Sale Report.csv",low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y')

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

print(df['Year'].nunique())
print(df['Month'].nunique())
print(df['Day'].nunique())
df=df.drop(columns=['Year','Date'])
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(24, 24))

# Order Status Distribution
sns.countplot(data=df, y='Status', ax=axes[0, 0])
axes[0, 0].set_title('Order Status Distribution')
axes[0, 0].set_xlabel('Number of Orders')
axes[0, 0].set_ylabel('Status')
axes[0, 0].tick_params(axis='y', rotation=0)

# Distribution of Size
sns.countplot(x=df['Size'], order=df['Size'].value_counts().index, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Size')
axes[0, 1].set_xlabel('Size')
axes[0, 1].set_ylabel('Count')

# Distribution of Courier Status
sns.countplot(x=df['Courier Status'], order=df['Courier Status'].value_counts().index, ax=axes[0, 2])
axes[0, 2].set_title('Distribution of Courier Status')
axes[0, 2].set_xlabel('Courier Status')
axes[0, 2].set_ylabel('Count')

# Regional Sales Distributions
top_states = df['ship-state'].value_counts().nlargest(10)
sns.barplot(x=top_states.index, y=top_states.values, ax=axes[1, 0])
axes[1, 0].set_title('Regional Sales Distributions')
axes[1, 0].set_xlabel('State')
axes[1, 0].set_ylabel('Number of Orders')
axes[1, 0].tick_params(axis='x', rotation=45)

# Fulfilment Type Distribution
sns.countplot(data=df, x='Fulfilment', ax=axes[1, 1])
axes[1, 1].set_title('Fulfilment Type Distribution')
axes[1, 1].set_xlabel('Fulfilment Type')
axes[1, 1].set_ylabel('Number of Orders')
axes[1, 1].tick_params(axis='x', rotation=45)

# Shipping Service Level Distribution
sns.countplot(data=df, x='ship-service-level', ax=axes[1, 2])
axes[1, 2].set_title('Shipping Service Level Distribution')
axes[1, 2].set_xlabel('Shipping Service Level')
axes[1, 2].set_ylabel('Number of Orders')
axes[1, 2].tick_params(axis='x', rotation=45)

# Sales Trends Over Time
sns.countplot(data=df, x='Month', hue='Courier Status', ax=axes[2, 0])
axes[2, 0].set_title('Sales Trends Over Time')
axes[2, 0].set_xlabel('Month')
axes[2, 0].set_ylabel('Number of Orders')
axes[2, 0].tick_params(axis='x', rotation=45)
axes[2, 0].legend(title='Status')

# Distribution of Amount
sns.histplot(df['Amount'], kde=True, bins=30, ax=axes[2, 1])
axes[2, 1].set_title('Distribution of Amount')
axes[2, 1].set_xlabel('Amount')
axes[2, 1].set_ylabel('Frequency')

# Distribution of Qty
sns.histplot(df['Qty'], kde=True, bins=30, ax=axes[2, 2])
axes[2, 2].set_title('Distribution of Qty')
axes[2, 2].set_xlabel('Qty')
axes[2, 2].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

### Step 4: Predictive Modeling
data["Courier Status"].value_counts()
# Due to the big diffrence between the classes in our targer Feature we will use a resampling technique called SMOTE
x = cleaned_data.drop(["Courier Status"], axis=1)
y = cleaned_data["Courier Status"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(x, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Test Set Metrics:")
y_pred_test = model.predict(X_test)

conf_matrix_test = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix (Test):")
ConfusionMatrixDisplay(conf_matrix_test).plot()
plt.title("Confusion Matrix - Test Set")
plt.show()

precision_test = precision_score(y_test, y_pred_test, average='macro')
recall_test = recall_score(y_test, y_pred_test, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

print(f"Precision (Test): {precision_test:.4f}")
print(f"Recall (Test): {recall_test:.4f}")
print(f"F1-score (Test): {f1_test:.4f}")

print("\n-----------------------------------------------------------------\n")

print("Train Set Metrics:")
y_pred_train = model.predict(X_train)

conf_matrix_train = confusion_matrix(y_train, y_pred_train)
print("Confusion Matrix (Train):")
ConfusionMatrixDisplay(conf_matrix_train).plot()
plt.title("Confusion Matrix - Train Set")
plt.show()

precision_train = precision_score(y_train, y_pred_train, average='macro')
recall_train = recall_score(y_train, y_pred_train, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')

print(f"Precision (Train): {precision_train:.4f}")
print(f"Recall (Train): {recall_train:.4f}")
print(f"F1-score (Train): {f1_train:.4f}")

# ```json
# From the above result we conclude that The logistic regression model exhibits high precision, recall, and F1-score on both the training and test datasets, indicating excellent performance and minimal overfitting. The close alignment of performance metrics between the training and test sets further confirms the model's generalization capability.
# ```
test_log_loss = log_loss(y_test, model.predict_proba(X_test))
print("Test Log Loss:", test_log_loss)

train_log_loss = log_loss(y_train, model.predict_proba(X_train))
print("Train Log Loss:", train_log_loss)
# The Loss diffrence is too low between train and test so there is no overfitting
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring_metrics)

print("Cross-Validation Results:")
for fold_idx in range(5):
    print(f"Fold {fold_idx + 1}:")
    print(f"Accuracy  : {cv_results['test_accuracy'][fold_idx]:.4f}")
    print(f"Precision : {cv_results['test_precision_macro'][fold_idx]:.4f}")
    print(f"Recall    : {cv_results['test_recall_macro'][fold_idx]:.4f}")
    print(f"F1-score  : {cv_results['test_f1_macro'][fold_idx]:.4f}")
    print()

print("Mean Scores:")
print(f"Mean Accuracy  : {cv_results['test_accuracy'].mean():.4f}")
print(f"Mean Precision : {cv_results['test_precision_macro'].mean():.4f}")
print(f"Mean Recall    : {cv_results['test_recall_macro'].mean():.4f}")
print(f"Mean F1-score  : {cv_results['test_f1_macro'].mean():.4f}")
# From the 5 folds we concluded that the stabillity of the training process across diffrent folds
### Step 5  Dashboard Development
# Grouping data for second dashboard
orders_per_day_month = df.groupby(['Month', 'Day']).size().reset_index(name='Number of Orders')

# Define your colors
colors = {
    'background': '#f9f9f9',
    'text': '#333333',
    'accent_color': '#1f77b4',
    'accent_color2': '#ff7f0e',
    'accent_color3': '#2ca02c',
    'grid_color': '#dddddd',
}

# List of colors to use for the lines
line_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A1FF33', '#5733FF']

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container(
    fluid=True,
    style={'backgroundColor': colors['background'], 'padding': '20px'},
    children=[
        dbc.Row(
            dbc.Col(
                html.H1(
                    children='Sales Dashboard',
                    style={'textAlign': 'center', 'color': colors['text']}
                )
            )
        ),
        
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id='graph1',
                    figure={
                        'data': [
                            {'x': orders_per_day_month[orders_per_day_month['Month'] == month]['Day'],
                             'y': orders_per_day_month[orders_per_day_month['Month'] == month]['Number of Orders'],
                             'type': 'line', 'name': f'Month {month}',
                             'line': {'color': line_colors[i % len(line_colors)]}}
                            for i, month in enumerate(orders_per_day_month['Month'].unique())
                        ],
                        'layout': {
                            'title': 'Number of Orders by Day for Each Month',
                            'xaxis': {'title': 'Day of the Month', 'gridcolor': colors['grid_color']},
                            'yaxis': {'title': 'Number of Orders', 'gridcolor': colors['grid_color']},
                            'plot_bgcolor': colors['background'],
                            'paper_bgcolor': colors['background'],
                            'font': {'color': colors['text']}
                        }
                    }
                ),
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Label("Select Month"),
                width=12,
                style={'marginBottom': '20px'}
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id='month-dropdown',
                    options=[{'label': month, 'value': month} for month in df['Month'].unique()],
                    value=df['Month'].unique()[0],
                    multi=False,
                    clearable=False,
                    style={'marginBottom': '20px'}
                ),
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(id='state-sales-graph'),
                width=12
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(id='city-sales-graph'),
                width=12
            )
        )
    ]
)

# Callback to update the graphs based on the selected month
@app.callback(
    [Output('state-sales-graph', 'figure'),
     Output('city-sales-graph', 'figure')],
    [Input('month-dropdown', 'value')]
)
def update_graphs(selected_month):
    filtered_df = df[df['Month'] == selected_month]

    state_sales = filtered_df.groupby('ship-state')['Amount'].sum().reset_index()
    top_state_sales = state_sales.nlargest(10, 'Amount')

    city_sales = filtered_df.groupby('ship-city')['Amount'].sum().reset_index()
    top_city_sales = city_sales.nlargest(10, 'Amount')

    state_fig = px.bar(top_state_sales, x='ship-state', y='Amount', title='Top 10 States by Sales',
                       color_discrete_sequence=[colors['accent_color']])
    city_fig = px.bar(top_city_sales, x='ship-city', y='Amount', title='Top 10 Cities by Sales',
                      color_discrete_sequence=[colors['accent_color2']])

    state_fig.update_layout(plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font={'color': colors['text']})
    city_fig.update_layout(plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font={'color': colors['text']})

    return state_fig, city_fig

if __name__ == '__main__':
    app.run_server(debug=True)
