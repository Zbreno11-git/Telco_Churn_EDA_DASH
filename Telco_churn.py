import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

file = '/Users/luanabreno/Desktop/Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv'

df=pd.read_csv(file)
print(df.head())
print(df.describe())
print(df.info())

print(df['Contract'].value_counts(normalize=True))
"""
Month-to-month    55%
Two year          24%
One year          20%
"""
# More than half of the clients have a Month-to-month contract
churn_by_contract = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
print(churn_by_contract)
"""
Churn                 No       Yes
Contract                          
Month-to-month        57%      43%
One year              88%      11%
Two year              97%      2.8%
"""
# Month to month clients have a higher churn rate
sns.heatmap(churn_by_contract, annot=True, cmap='YlGnBu')
plt.yticks(rotation=0)
plt.title('Churn by Contract')
plt.tight_layout()
plt.savefig('churn_contract.png')

client_profile = df.groupby(['InternetService', 'TechSupport'])['Churn'].value_counts(normalize=True).unstack()
print(client_profile)

# Clients with Fiber Optic WITHOUT Tech Support cancel the most... ~49%

"""
Churn                                  No   Yes
InternetService TechSupport                    
DSL             No                    898   345
                Yes                  1064   114
Fiber optic     No                   1129  1101
                Yes                   670   196
No              No internet service  1413   113
"""

client_profile.plot(kind='bar', figsize=(11,7))
plt.title('Churn by Internet Service')
plt.xticks(rotation=0)
plt.xlabel('(InternetService, TechSupport)')
plt.tight_layout()
plt.savefig('churn_internetserv.png')

fiber_mask = df['InternetService'] == 'Fiber optic'
print(df[fiber_mask]['MonthlyCharges'].mean())

# The average monthly charge for Fiber optic is $91.50
print(91.50 * 1101)
# Losing a 1101 clients due to Tech Support costs $100.741.5 per month to the company
# INCREASE TECH SUPPORT FOR FIBER OPTIC CLIENTS

top_10_clients_total = (df[['tenure', 'Contract', 'MonthlyCharges', 'Churn', 'TotalCharges']]
                  .sort_values(ascending=False, by='TotalCharges').head(10))
print(top_10_clients_total)
# From my top 10 (by Total Charges) I have TWO cancellations  - let's take a look:
print(df.iloc[3686])
"""
customerID              5899-MQZZL
gender                      Female
SeniorCitizen                    0
Partner                         No
Dependents                      No
tenure                          13
PhoneService                   Yes
MultipleLines                  Yes
InternetService        Fiber optic *** 
OnlineSecurity                  No
OnlineBackup                    No
DeviceProtection                No
TechSupport                     No ---> lack of tech support
StreamingTV                     No
StreamingMovies                 No
Contract            Month-to-month
PaperlessBilling               Yes
PaymentMethod         Mailed check
MonthlyCharges                75.0
TotalCharges                999.45
Churn                          Yes
"""
print(df.iloc[6179])
"""
customerID                         6328-ZPBGN
gender                                 Female
SeniorCitizen                               1
Partner                                    No
Dependents                                 No
tenure                                     11
PhoneService                              Yes
MultipleLines                             Yes
InternetService                   Fiber optic ***
OnlineSecurity                             No
OnlineBackup                               No
DeviceProtection                           No
TechSupport                                No ---> lack of tech support
StreamingTV                               Yes
StreamingMovies                           Yes
Contract                       Month-to-month
PaperlessBilling                          Yes
PaymentMethod       Bank transfer (automatic)
MonthlyCharges                          95.15
TotalCharges                           997.65
Churn                                     Yes
"""

top_10_clients_tenure = (df[['tenure', 'Contract', 'MonthlyCharges', 'Churn', 'TotalCharges']]
                  .sort_values(ascending=False, by='tenure').head(10))
print(top_10_clients_tenure)

# From my top 10 (by tenure) I have ONE cancellation  - let's take a look:
print(df.iloc[3111])

"""
customerID                8809-RIHDD
gender                          Male
SeniorCitizen                      0
Partner                          Yes
Dependents                       Yes
tenure                            72
PhoneService                     Yes
MultipleLines                    Yes
InternetService          Fiber optic
OnlineSecurity                    No
OnlineBackup                     Yes
DeviceProtection                 Yes
TechSupport                       No ---> wtf is going on with the IT team
StreamingTV                      Yes
StreamingMovies                  Yes
Contract                    Two year
PaperlessBilling                 Yes
PaymentMethod       Electronic check
MonthlyCharges                 103.4
TotalCharges                 7372.65
Churn                            Yes
"""

# List of customerID's to reach and offer Tech Support
clients_to_rescue = df[(df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 'No')]['customerID'].reset_index()
print(f'Number of clients to reach: {clients_to_rescue.count()} (high chance of churn)')
    # 2230 Clients with high risk of churn
#print(df.loc[df['InternetService'] == 'Fiber optic', 'MonthlyCharges'].mean())
    # Knowing that the average monthly charge of the fiber is $91.50
    # Total Revenue Exposure: $204.045/month
print(clients_to_rescue.head(10))

# Logistic Regression - Predict Churn
    # Transform key datas

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges']= df['TotalCharges'].fillna(df['TotalCharges'].median())

strong_features = [
    'Contract',
    'MonthlyCharges',
    'TotalCharges',
    'tenure',
    'InternetService',
    'OnlineSecurity',
    'PaymentMethod',
    'PaperlessBilling',
    'PhoneService',
    'TechSupport'
]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

X = df[strong_features]
y = df['Churn'].map({'Yes': 1, 'No': 0})

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

proba = rf.predict_proba(X_test)[:, 1]

threshold = 0.45
pred = (proba > threshold).astype(int)

print(f'AUC: {roc_auc_score(y_test, proba)}')
print(confusion_matrix(y_test, pred))
"""
AUC: 0.8576
	1141 (TN)	398 (FP)
	100 (FN)	474 (TP)
 """

# Average monthly charge for those who are cancelling
print(df[df['Churn'] == 'Yes']['MonthlyCharges'].mean())
    # $74.44
# By identifying 474 high-risk customers, the model enables targeted retention actions,...
# ...representing up to ~$35K/month in recoverable revenue


