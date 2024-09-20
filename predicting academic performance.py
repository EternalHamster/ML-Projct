import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#Reading the data from csv file and removing unwanted properties
df = pd.read_csv('StudentPerformanceFactors.csv')
df.drop(columns=['Attendance', 'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', \
                 'Previous_Scores', 'Motivation_Level', 'Internet_Access', 'Tutoring_Sessions', 'Family_Income', \
                'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', \
                'Distance_from_Home', 'Gender'], inplace=True)

#Normalizing the features values
df['Hours_Studied_N'] = (df['Hours_Studied'] - df['Hours_Studied'].min()) / (df['Hours_Studied'].max() - df['Hours_Studied'].min())
df['Sleep_Hours_N'] = (df['Sleep_Hours'] - df['Sleep_Hours'].min()) / (df['Sleep_Hours'].max() - df['Sleep_Hours'].min())
df['Physical_Activity_N'] = (df['Physical_Activity'] - df['Physical_Activity'].min()) / (df['Physical_Activity'].max() - df['Physical_Activity'].min())


#Adding the features together without any weight and normalizing the feature
df['Feature'] = 4*df['Hours_Studied_N'] + df['Physical_Activity_N'] + 2*df['Sleep_Hours_N']
df['Feature_N'] = (df['Feature'] - df['Feature'].min()) / (df['Feature'].max() - df['Feature'].min())

#Normalizing the label
df['Exam_Score_N'] = (df['Exam_Score'] - df['Exam_Score'].min()) / (df['Exam_Score'].max() - df['Exam_Score'].min())

#Dropping the unused columns
df.drop(columns=['Hours_Studied', 'Hours_Studied_N', 'Sleep_Hours', 'Sleep_Hours_N', 'Physical_Activity', \
                 'Physical_Activity_N', 'Feature'], inplace=True)

X = df['Feature_N'].to_numpy().reshape(-1,1)
y = df['Exam_Score_N'].to_numpy()

plt.scatter(X, y, color='blue')

plt.xlabel('X')
plt.ylabel('y')
plt.show()