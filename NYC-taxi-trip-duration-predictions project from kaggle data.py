#!/usr/bin/env python
# coding: utf-8

# This project aims to predict the duration of taxi trips in New York City using data from the 2016 Kaggle competition. We perform data exploration, feature engineering, model building, and evaluation.

# In[39]:


import numpy as np 
import pandas as pd


# ## 1. Data Overview
# 
# Let's load the dataset and examine its basic structure, size, and features.

# In[40]:


train = pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train


# ## 2. Exploratory Data Analysis (EDA) and Feature Engineering
# 
# In this section, we explore the dataset to identify patterns, detect outliers, and gain insights that may help improve the model's performance.We create new features from existing ones to enrich the dataset and provide more useful information for our model.

# In[41]:


train.isnull().sum()


# In[42]:


test.isnull().sum()


# In[43]:


train.duplicated().sum()


# In[44]:


test.duplicated().sum()


# In[45]:


train.describe()


# In[46]:


train.dtypes


# In[47]:


train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['dropoff_datetime']=pd.to_datetime(train['dropoff_datetime'])
train['Day'] = train['pickup_datetime'].dt.dayofweek
train['Month'] = train['pickup_datetime'].dt.month

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
test['Day'] = test['pickup_datetime'].dt.dayofweek
test['Month'] = test['pickup_datetime'].dt.month

train['pickup_weekday'] = train['pickup_datetime'].dt.weekday 
train['is_weekend'] = train['pickup_weekday'].isin([5, 6]).astype(int)
train.drop('pickup_weekday',axis=1,inplace=True)

train['pickup_hour'] = train['pickup_datetime'].dt.hour
train['is_rush_hour'] = train['pickup_hour'].isin([7, 8, 9,10,11,12,13,14,15, 16, 17, 18, 19,20,21,22]).astype(int)


test['pickup_weekday'] = test['pickup_datetime'].dt.weekday
test['is_weekend'] = test['pickup_weekday'].isin([5, 6]).astype(int)
test.drop('pickup_weekday',axis=1,inplace=True)

test['pickup_hour'] = test['pickup_datetime'].dt.hour
test['is_rush_hour'] = test['pickup_hour'].isin([7, 8, 9,10,11,12,13,14,15, 16, 17, 18, 19,20,21,22]).astype(int)


# In[48]:


get_ipython().system('pip install haversine')
from haversine import haversine

def compute_distance(row):
    pickup = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff = (row['dropoff_latitude'], row['dropoff_longitude'])
    return haversine(pickup, dropoff)

train['distance_km'] = train.apply(compute_distance, axis=1)
test['distance_km']=test.apply(compute_distance,axis=1)


# In[49]:


train.drop(columns=['pickup_longitude','dropoff_longitude','pickup_latitude','dropoff_latitude'],inplace=True)
test.drop(columns=['pickup_datetime'],inplace=True)
train.drop(columns=['pickup_datetime','dropoff_datetime'],inplace=True)
test.drop(columns=['pickup_longitude','dropoff_longitude','pickup_latitude','dropoff_latitude'],inplace=True)  


# In[50]:


train['store_and_fwd_flag']=train['store_and_fwd_flag'].replace({'N':1,'Y':2})
test['store_and_fwd_flag']=test['store_and_fwd_flag'].replace({'N':1,'Y':2})


# In[51]:


from itertools import combinations
train1=train.copy().drop(columns=['id','trip_duration'])
numeric_cols = train1.select_dtypes(include='number').columns

for col1, col2 in combinations(numeric_cols, 2):
    new_col_name = f'{col1}_x_{col2}'
    train[new_col_name] = train[col1] * train[col2]
    test[new_col_name]=test[col1]*test[col2]


# ## VISUAL PRESENTATION OF FEATURES

# In[52]:


import matplotlib.pyplot as plt 
import seaborn as sns

plt.figure(figsize=(12,6))
sns.countplot(data=train, x='pickup_hour', palette='viridis')
plt.title('Frequency of taxi trips per hour (Rush Hours)')
plt.xlabel('Hour of day')
plt.ylabel('Number of trips')
plt.xticks(range(0,24))
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()


# In[53]:


day_names = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
train['DayName'] = train['Day'].map(dict(enumerate(day_names)))
plt.figure(figsize=(10,6))
sns.countplot(data=train, x='DayName', order=day_names, palette='magma')
plt.title('📅 Frequency of taxi trips per day')  
plt.xlabel('Day')
plt.ylabel('Number of trips')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()


# In[54]:


plt.figure(figsize=(10, 6))
sns.countplot(data=train, x='vendor_id', palette='viridis')
plt.title('📊 Trips per vendor_id')
plt.xlabel('Vendor ID')
plt.ylabel('Number of trips')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()


# In[55]:


train.drop('DayName',axis=1,inplace=True)
train2=train.copy().drop(columns=['id'],axis=1) 

train2.hist(bins=30, figsize=(20, 12), edgecolor='black') 
plt.suptitle("Distributions of Features", fontsize=16) 
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10)) 
sns.heatmap(train2.corr(), cmap='coolwarm', annot=False) 
plt.title("Correlation Heatmap", fontsize=14) 
plt.show()

target = 'trip_duration'
if target in train2.columns: corr_target = train2.corr()[target].sort_values(ascending=False) 
print("\nCorrelation with target:") 
print(corr_target)


# ## 4. Model Building
# 
# We train different machine learning models to predict trip duration. We also tune hyperparameters and evaluate model performance.

# In[56]:


get_ipython().system('pip install xgboost')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

num_cols = train.select_dtypes(include=['int64', 'float64']).columns
target_col='trip_duration'
masks = []
for col in num_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    masks.append((train[col] >= lower) & (train[col] <= upper))

combined_mask = np.logical_and.reduce(masks)
train_clean = train[combined_mask].copy()
train_full = train.copy()

print(f"Χωρίς outliers: {train_clean.shape}, Με outliers: {train_full.shape}")


features = train_clean.columns.drop(['id', 'trip_duration'])

def prepare_data(df):
    X = df[features]
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_clean, X_test_clean, y_train_clean, y_test_clean = prepare_data(train_clean)


scaler = StandardScaler()
X_train_clean_scaled = scaler.fit_transform(X_train_clean)
X_test_clean_scaled = scaler.transform(X_test_clean)


models = {
    'Linear (clean)': LinearRegression().fit(X_train_clean_scaled, y_train_clean),
    'XGB (clean)': XGBRegressor(n_estimators=100, random_state=42).fit(X_train_clean, y_train_clean),
}

for name, model in models.items():
    if 'Linear' in name:
        X_test = X_test_clean_scaled 
        y_test = y_test_clean 
    else:
        X_test = X_test_clean 
        y_test = y_test_clean 

    preds = model.predict(X_test)
    print(f"\n🔹 {name}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"RMSE: {mean_squared_error(y_test, preds, squared=False):.2f}")
    print(f"R²: {r2_score(y_test, preds):.4f}")
    


# In[57]:


get_ipython().system('pip install shap')
import shap

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train_clean, y_train_clean)


explainer = shap.Explainer(model)


X_sample = X_train_clean.sample(100000, random_state=42)
shap_values = explainer(X_sample)

shap.summary_plot(shap_values, X_sample, max_display=X_sample.shape[1])


# ## 5. Model Evaluation
# 
# We evaluate our models using appropriate metrics and analyze the results to select the best-performing model.

# In[58]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(model, X_train_clean, y_train_clean, scoring='neg_root_mean_squared_error', cv=5)
print(f"Cross-validated RMSE: {-np.mean(scores):.2f}")


# In[59]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.05],
}

grid_search = GridSearchCV(XGBRegressor(random_state=42),
                           param_grid,
                           scoring='neg_root_mean_squared_error',
                           cv=3,
                           verbose=1,
                           n_jobs=-1)

grid_search.fit(X_train_clean, y_train_clean)

print(f"Best Params: {grid_search.best_params_}")
print(f"Best RMSE (log-scale): {-grid_search.best_score_:.2f}")


# In[60]:


best_model = grid_search.best_estimator_
preds = best_model.predict(X_test_clean)

final_preds = preds
actuals = y_test_clean
print(f"Final MAE: {mean_absolute_error(actuals, final_preds):.2f}")
print(f"Final RMSE: {mean_squared_error(actuals, final_preds, squared=False):.2f}")
print(f"Final R²: {r2_score(actuals, final_preds):.4f}")


# SELECTION OF THE FIRST 25 FEATURES WITH THE MOST IMPACT ON THE PREDICTIONS AS WE SEE FROM SHAP VALUES.

# In[61]:


X_train_clean=X_train_clean.iloc[:,:25]
X_train_clean


# In[62]:


best_model = grid_search.best_estimator_
preds = best_model.predict(X_test_clean)


# In[63]:


param_grid = {
'n_estimators': [50, 100],
'max_depth': [3, 5, 7],
'learning_rate': [0.1, 0.05],
}
grid_search = GridSearchCV(XGBRegressor(random_state=42),
 param_grid,
 scoring='neg_root_mean_squared_error',
 cv=3,
 verbose=1,
 n_jobs=-1)
grid_search.fit(X_train_clean, y_train_clean)
print(f"Best Params: {grid_search.best_params_}")
print(f"Best RMSE : {-grid_search.best_score_:.2f}")


# In[64]:


X_test_clean=X_test_clean.iloc[:,:25]
best_model2 = grid_search.best_estimator_
new_preds = best_model2.predict(X_test_clean)

print(f'MAE IS:{mean_absolute_error(new_preds,y_test_clean)}')


# In[65]:


ids=test['id']
test.drop(columns=['id'],axis=1,inplace=True)
test=test.iloc[:,:25]
sub_preds=best_model2.predict(test)
sub_preds


# In[66]:


submission=pd.DataFrame({'id':ids,'trip_duration':sub_preds})
submission.to_csv('submission.csv',index=False)

