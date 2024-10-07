import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error

#Reading the data from csv file and removing unwanted properties
df = pd.read_csv('StudentPerformanceFactors.csv')
df.drop(columns=['Attendance', 'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', \
                 'Previous_Scores', 'Motivation_Level', 'Internet_Access', 'Tutoring_Sessions', 'Family_Income', \
                'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', \
                'Distance_from_Home', 'Gender'], inplace=True)
df.dropna(inplace=True)

#Normalizing the features values
df['Hours_Studied_N'] = (df['Hours_Studied'] - df['Hours_Studied'].min()) / (df['Hours_Studied'].max() - df['Hours_Studied'].min())
df['Sleep_Hours_N'] = (df['Sleep_Hours'] - df['Sleep_Hours'].min()) / (df['Sleep_Hours'].max() - df['Sleep_Hours'].min())
df['Physical_Activity_N'] = (df['Physical_Activity'] - df['Physical_Activity'].min()) / (df['Physical_Activity'].max() - df['Physical_Activity'].min())

#Adding the features together without any weight and normalizing the feature
df['Feature'] = 4*df['Hours_Studied_N'] + df['Physical_Activity_N'] + 2*df['Sleep_Hours_N']
df['Feature_N'] = (df['Feature'] - df['Feature'].min()) / (df['Feature'].max() - df['Feature'].min())

#Normalizing the label
df['Exam_Score_N'] = (df['Exam_Score'] - df['Exam_Score'].min()) / (df['Exam_Score'].max() - df['Exam_Score'].min())



""" RANDOMFOREST """

df = df[['Hours_Studied', 'Sleep_Hours', 'Physical_Activity', 'Exam_Score']].dropna()


df_scaled = MinMaxScaler().fit_transform(df)

#Clustering with DBSCAN to eliminate outliers
dbscan = DBSCAN(eps=0.2, min_samples=50).fit(df_scaled)

scatter = plt.scatter(df_scaled[:, 0], df_scaled[:, 3], c=dbscan.labels_)
plt.title('DBSCAN Clusters')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.show()

df_noliars = df[dbscan.labels_ != -1].reset_index(drop=True)
scatter = plt.scatter(df_noliars['Hours_Studied'], df_noliars['Exam_Score'])
plt.title('Post-DBSCAN')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.show()

X = df_noliars[['Hours_Studied', 'Sleep_Hours', 'Physical_Activity']]
y = df_noliars['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

rfrm = RandomForestRegressor()

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [5, 10, 20]
}

# Set up gridsearch with k-fold cross-validation of n_splits=5
grid_search = GridSearchCV(rfrm, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)

#Make predictions using the best model found
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

#Evaluating the model
tr_error = mean_squared_error(y_test, y_pred)
print(f"Training error: {tr_error}")

val_score = cross_val_score(rfrm, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
val_error = -np.mean(val_score)
print(f"Validation error: {val_error}")

#Plotting REAL VS. PREDICTED
plt.scatter(y_test, y_pred, color='blue')
plt.plot([55, 85], [55, 85]) 
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Actual vs. Predicted Exam Scores')
plt.show()

#Plotting the importances
feature_importances = best_model.feature_importances_
features = ['Hours Studied', 'Sleep Hours', 'Physical Activity']
plt.barh(features, feature_importances, color='red')
plt.xlabel('Feature Importance (weight)')
plt.title('Feature Significance in Predicting Exam Score')
plt.show()
