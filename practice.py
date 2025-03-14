import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from imblearn.over_sampling import SMOTE
# from imblearn.combine import SMOTEENN
# from imblearn.under_sampling import RandamUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, classification_report,f1_score
)

#pip install imbalanced-learn
#pip install scikit-learn

df = pd.read_csv("DataSet1/Iris.csv")

# print("statistics:")
# mean_sepalwidth = df['SepalWidthCm'].mean()
# mediansw = df['SepalWidthCm'].median()
# modesw = df['SepalWidthCm'].mode()[0]
# sd = df['SepalWidthCm'].std()
# rangesw = df['SepalWidthCm'].max() - df['SepalWidthCm'].min()
# print(f"Mean:{mean_sepalwidth} \nMedian:{mediansw} \nmode:{modesw} \nStandard_deviation:{sd} \nrange:{rangesw}")


# print("Missing Values Before Handling: ",df.isnull().sum())
df['SepalWidthCm'].fillna(df['SepalWidthCm'].median(),inplace=True)
df['SepalLengthCm'].fillna(df['SepalLengthCm'].median(),inplace=True)
df['PetalLengthCm'].fillna(df['PetalLengthCm'].median(),inplace=True)
df['PetalWidthCm'].fillna(df['PetalWidthCm'].median(),inplace=True)
# print("Missing Values After Handling:",df.isnull().sum())


x = df.drop("Species",axis=1)
y = df['Species']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=42)

dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
predictions = dt_model.predict(x_test)
print("Metrics: ")
print(f"Accuracy : {accuracy_score(y_test,predictions)}")
print(f"precision : {precision_score(y_test,predictions)}")
print(f"recall : {recall_score(y_test,predictions)}")
print(f"F1-score : {f1_score(y_test, predictions)}")
print(f"confusion Matrix : \n{classification_report(y_test,predictions)}")





# x = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
# y = df['Species']
# print("Original class distribution : ",Counter(y))

# #SMOTE
# smote = SMOTE(radom_state=42)
# x_smote,y_smote = smote.fit_resample(x,y)
# print("After SMOTE:",Counter(y_smote))

# #underSampling
# undersampler = RandamUnderSampler(random_state=42)
# x_under,y_under = undersampler.fit_resample(x,y)
# print("After UnderSampling:",Counter(y_under))







# print("Scaling")
# std_scaler = StandardScalar()
# df['petalLengthCm_scaled'] = std_scaler.fit_transform(std_scaler.df[['petalLengthCm']])
# print(df[['petalLengthCm','petalLengthCm_scaled']].head())






# print("correlation and Covariance: \n")
# numcols = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
# df_numeric = df[numcols]

# df_numeric.to_csv("/species_classification.csv",index=False)
# corr_mat,cov_mat = df_numeric.corr(),df_numeric.cov()

# # HeatMap for correlation
# plt.figure(figsize=(8,6))
# sns.heatmap(corr_mat,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
# plt.title("HeatMap of Correlation Matrix")
# plt.show()

# # HeatMap for covariance
# plt.figure(figsize=(8,6))
# sns.heatmap(cov_mat,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
# plt.title("HeatMap of Covairance Matrix")
# plt.show()







