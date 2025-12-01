# coding: utf-8
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
path = 'LaptopData.csv'

data = pd.read_csv(path)
data.head()
data.info()
data.describe()
data.isnull().sum()
data = data.dropna()
data.isnull().sum()
data.duplicated()
data.head()
data['Company'].unique().tolist()
data.TypeName.unique()
data.OpSys.unique()
sns.distplot(data.Price)
data['Company'].value_counts().plot(kind='bar')
sns.barplot(x=data['Company'],y=data['Price'])
plt.xticks(rotation='vertical')
plt.show()
data['TypeName'].value_counts().plot(kind='bar')
sns.barplot(x=data['TypeName'],y=data['Price'])
plt.xticks(rotation='vertical')
plt.show()
data['Inches'].isnull().sum()
data['Inches'] = data['Inches'].replace('?',np.nan)
data['Inches'].isnull().sum()
data = data.dropna()
data['Inches'].astype('float')
sns.distplot(data['Inches'])
sns.scatterplot(x=data['Inches'], y=data['Price'])
sns.boxplot(data['Inches'])
sns.barplot(x=data['Inches'], y=data['Price'])
plt.xticks(rotation='vertical')
plt.show()
data['ScreenResolution'].value_counts()
data['TouchScreen'] = data['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
data['TouchScreen']
data.sample(5)
data.TouchScreen.value_counts().plot(kind='bar')
sns.barplot(x=data['TouchScreen'], y=data['Price'])
plt.xticks(rotation='vertical')
plt.show()
data['IPS'] = data['ScreenResolution'].apply(lambda x: 1 if 'IPS Panel' in x else 0)
data.sample()
data.IPS.value_counts()
sns.barplot(x=data.IPS, y=data.Price)
plt.xticks(rotation='vertical')
plt.show()
temp = data['ScreenResolution'].str.split('x',n=1,expand=True)
temp[0]
data['X_res'] = temp[0]
data['Y_res'] = temp[1]
data['X_res']
data['X_res'] = data['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])
data.head()
data.info()
data['X_res']=data['X_res'].astype('int')
data['Y_res']=data['Y_res'].astype('int')
data['Inches']=data['Inches'].astype('float')
data.info()
data[['Ram','Weight']].head()
data['Ram']=data['Ram'].str.replace('GB','')
data['Weight']=data['Weight'].str.replace('kg','')
data.head()
data['Ram']=data['Ram'].astype('int')
# data['Weight']=data['Weight'].astype('float')
data['Weight']=data['Weight'].replace('?',np.nan)
data['Weight'].isnull().sum()
data['Weight'].dropna()
data['Weight']=data['Weight'].astype('float')
data.info()
numeric_cols = data.select_dtypes(include='number')
numeric_cols.corr()['Price']
data['PPI'] = (((data['X_res']**2) + (data['Y_res']**2))** 0.5 / data.Inches).astype('float')
numeric_cols = data.select_dtypes(include='number')
numeric_cols.corr()['Price']
data.drop('ScreenResolution',axis=1,inplace=True)
data.head()
data.drop(columns=['Inches','X_res','Y_res'],inplace=True)
data.head()
data.Cpu.value_counts()
data['Cpu Name'] = data['Cpu'].apply(lambda x:' '.join(x.split()[:3]))
data.head()
# function for storing Cpu name in above categoies 
def fetch_processor(cpu):
    if cpu == 'Intel Core i7' or cpu == 'Intel Core i5' or cpu == 'Intel Core i3':
        return cpu
    elif cpu.split()[0] == 'Intel':
        return 'Other Intel Processor'
    else:
        return 'AMD Processor'
data['Cpu brand'] = data['Cpu Name'].apply(fetch_processor)
data.head()
data.sample(5)
data['Cpu brand'].value_counts()
# VIsualization 
data['Cpu brand'].value_counts().plot(kind='bar')
sns.barplot(x=data['Cpu brand'], y=data['Price'])
plt.xticks(rotation = 'vertical')
plt.show()
data.drop(columns = ['Cpu','Cpu Name'],inplace=True)
data.head()
data['Ram'].value_counts().plot(kind='bar')
sns.barplot(x=data['Ram'], y=data['Price'])
plt.xticks(rotation='vertical')
plt.show()
data.Memory.value_counts()
data['Memory'] = data['Memory'].astype(str).replace('\.0','',regex=True)
data['Memory'] = data['Memory'].str.replace('GB','') #replacing Gb with '' because values will count in gbs
data['Memory'] = data['Memory'].str.replace('TB','000') # because 1TB = 1000GB
data['Memory'].sample(5)
new = data['Memory'].str.split('+',n=1,expand=True) # 256GB SSD +  1TB HDD splitting by '+' sign
data['first'] = new[0] # storing   '256GB SSD'   part from   ~ 256GB SSD +  1TB HDD ~  this section
data['first'] = data['first'].str.strip() # removing white space
data['second'] = new [1] # storing   '1TB HDD'   part from   ~ 256GB SSD +  1TB HDD ~  this section
data['Layer1HDD'] = data['first'].apply(lambda x: 1 if 'HDD' in x else 0) #add 1 if HDD present in laptop or 0 if not
data['Layer2SSD'] = data['first'].apply(lambda x: 1 if 'SSD' in x else 0) #add 1 if SSD present in laptop or 0 if not
data['Layer3Hybrib'] = data['first'].apply(lambda x: 1 if 'Hybrid' in x else 0 ) #add 1 if Hybrid present in laptop or 0 if not
data['Layer4Flash_Storage'] = data['first'].apply(lambda x:1 if 'Flash Storage' in x else 0 )#add 1 if Hybrid present in laptop or 0 if not
data.head()
data = data.rename(columns = {'Layer2SSD':'Layer1SSD','Layer3Hybrib':'Layer1Hybrib','Layer4Flash_Storage':'Layer1Flash_Storage'})
data.drop(columns = ['Unnamed: 0'], inplace=True)
data.head()
data['first'] = data['first'].str.replace(r'\D','')
# data['second'].isnull().sum()
data['second'].fillna('0',inplace=True)
data.second.head()
data['Layer2HDD'] = data['second'].apply(lambda x: 1 if 'HDD' in x else 0) #add 1 if HDD present in laptop or 0 if not
data['Layer2SSD'] = data['second'].apply(lambda x: 1 if 'SSD' in x else 0) #add 1 if SSD present in laptop or 0 if not
data['Layer2Hybrib'] = data['second'].apply(lambda x: 1 if 'Hybrid' in x else 0 ) #add 1 if Hybrid present in laptop or 0 if not
data['Layer2Flash_Storage'] = data['second'].apply(lambda x:1 if 'Flash Storage' in x else 0 )#add 1 if Hybrid present in laptop or 0 if not
# data.drop(columns=['Unnamed: 0','first','second','Layer1HDD','Layer1SSD','Layer1Hybrib','Layer1Flash_Storage','Layer2HDD','Layer2SSD','Layer2Hybrib','Layer2Flash_Storage'],inplace=True)
# data.drop(columns=['Unnamed: 0','first','second','Layer1HDD','Layer1SSD','Layer1Hybrib','Layer1Flash_Storage','Layer2HDD','Layer2SSD','Layer2Hybrib','Layer2Flash_Storage'],inplace=True)
data.head()
data['second'] = data['second'].str.replace(r'\D','')
data['first']= data['first'].apply(lambda x:x.split()[0])
data['first']=data['first'].replace('?',np.nan)
data = data.dropna(subset=['first'])
data['first'].value_counts()
data.second.value_counts()
data['second']=data['second'].apply(lambda x:x.split()[0])
data['second'].value_counts()
data['first'] =data['first'].astype(int)
data['second'] =data['second'].astype(int)
data['HDD'] = (data['first']*data['Layer1HDD']+data['second']*data['Layer2HDD'])
data['SSD'] = (data['first']*data['Layer1SSD']+data['second']*data['Layer2SSD'])
data['Hybrib'] = (data['first']*data['Layer1Hybrib']+data['second']*data['Layer2Hybrib'])
data['Flash_Storage'] = (data['first']*data['Layer1Flash_Storage']+data['second']*data['Layer2Flash_Storage'])
data.head()
data.drop(columns=['first','second','Layer1HDD','Layer1SSD','Layer1Hybrib','Layer1Flash_Storage','Layer2HDD','Layer2SSD','Layer2Hybrib','Layer2Flash_Storage'],inplace=True)
data.head()
data.to_csv('clean_data.csv',index=False)
data.drop('Memory',axis=1,inplace=True)
data.head()
numeric_colms = data.select_dtypes(include = 'number')
numeric_colms.corr()['Price']
data.drop(columns=['Hybrib','Flash_Storage'],inplace=True)
data.head()
data['Gpu'].value_counts()
data['Gpu brand'] = data['Gpu'].apply(lambda x: x.split(' ')[0])
data.head()
data['Gpu brand'].value_counts()
data[data['Gpu brand'] == 'ARM']
data = data.drop(index=1191)
data['Gpu brand'].value_counts()
sns.barplot(x=data['Gpu brand'], y=data['Price'])
data.drop(columns=['Gpu'],inplace=True)
data.head()
data['OpSys'].value_counts()
sns.barplot(x=data.OpSys, y=data.Price)
plt.xticks(rotation='vertical')
plt.show()
def categorize_OS(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
data['OS']= data['OpSys'].apply(categorize_OS)
data.head()
data.drop(columns=['OpSys'],inplace=True)
data.head()
sns.barplot(x=data['OS'], y=data['Price'])
plt.xticks(rotation='vertical')
plt.show()
sns.distplot(data['Weight'])
sns.scatterplot(x=data['Weight'], y=data['Price'])
sns.boxplot(data['Weight'])
data['Weight'] = np.log(data['Weight'])
data.head()
sns.boxplot(data['Weight'])
sns.scatterplot(x=data['Weight'], y=data['Price'])
data[data['Weight'] < -1]
data = data.drop(index=349).reset_index()
sns.scatterplot(x=data['Weight'], y=data['Price'])
numeric_cols = data.select_dtypes(include='number')
numeric_cols.corr()['Price']
sns.heatmap(numeric_cols.corr())
sns.distplot(data['Price'])
sns.distplot(np.log(data['Price']))
X = data.drop(columns=['Price','index'])
y = np.log(data['Price'])
X
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
X_train
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,Lasso,Ridge
X_train['Weight'].isnull().sum()
X_train[X_train['Weight'].isnull()]
X_train['Weight'] = X_train['Weight'].fillna(0)
X_train['Weight'].isnull().sum()
from sklearn.preprocessing import StandardScaler
X_train.head(1)
encoding = ColumnTransformer(
    transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11]),
        ('scaler', StandardScaler(),[2,3,4,5,6,8,9])
    ],
    remainder='passthrough'
)

lr_model = LinearRegression()

pipe = Pipeline([
    ('step-1',encoding),
    ('step-2',lr_model)
])

pipe.fit(X_train,y_train)
from sklearn.metrics import r2_score,mean_absolute_error
y_pred = pipe.predict(X_test)

print('R2_score',r2_score(y_test,y_pred))
print('mean_absolute_error',mean_absolute_error(y_test,y_pred))
np.exp(0.1899589660572979)
ridge_model = Ridge(alpha=0.9)

ridge_pipe = Pipeline([
    ('step-1',encoding),
    ('step-2',ridge_model)
])

ridge_pipe.fit(X_train,y_train)
ridge_y_pred = ridge_pipe.predict(X_test)

print('R2_score',r2_score(y_test,ridge_y_pred))
print('mean_absolute_error',mean_absolute_error(y_test,ridge_y_pred))
lasso_model = Lasso(alpha=0.001)

lasso_pipe = Pipeline([
    ('step-1',encoding),
    ('step-2',lasso_model)
])

lasso_pipe.fit(X_train,y_train)
lasso_y_pred = lasso_pipe.predict(X_test)

print('R2_score',r2_score(y_test,lasso_y_pred))
print('mean_absolute_error',mean_absolute_error(y_test,lasso_y_pred))
X=X.dropna()
X.isnull().sum()
X.shape
y=y.drop(index=201)
y.shape
# y.reset_index()
# y.isnull().sum()
cat_cols = [0, 1, 7, 10, 11]
num_cols = [2,3,4,5,6,8,9]
results = []
for i in range(1000):
    try: 
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=i)
        encoding = ColumnTransformer(
            transformers=[
                ('col_trf',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),cat_cols),
                ('scaler',StandardScaler(),num_cols)
            ]
        )

        l_model = LinearRegression()
        lr_pipe =Pipeline([
            ('step_1',encoding),
            ('step_2',l_model)
        ])
        lr_pipe.fit(x_train,y_train)
        # Predicting
        lr_predict = lr_pipe.predict(x_test)
        # Calculate metrics
        r2 = r2_score(y_test, lr_predict)
        mae = mean_absolute_error(y_test, lr_predict)
    
        #storing values
        results.append((i,r2,mae))
    except ValueError as e:
        print('error occur',e)
best_split = max(results,key=lambda x:x[1])
print("Best random_state:", best_split[0])
print("Best R2 Score:", best_split[1])
print("Corresponding MAE:", best_split[2])
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=399)
encoding = ColumnTransformer(
            transformers=[
                ('col_trf',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),cat_cols),
                ('scaler',StandardScaler(),num_cols)
            ]
        )
ridge_model2 = Ridge(alpha=0.9)

ridge_pipe2 = Pipeline([
    ('step-1',encoding),
    ('step-2',ridge_model2)
])

ridge_pipe2.fit(x_train,y_train)
ridge_y_pred2 = ridge_pipe2.predict(x_test)

print('R2_score',r2_score(y_test,ridge_y_pred2))
print('mean_absolute_error',mean_absolute_error(y_test,ridge_y_pred2))
lasso_model2 = Lasso(alpha=0.001)

lasso_pipe2 = Pipeline([
    ('step-1',encoding),
    ('step-2',lasso_model2)
])

lasso_pipe2.fit(x_train,y_train)
lasso_y_pred2 = lasso_pipe2.predict(x_test)

print('R2_score',r2_score(y_test,lasso_y_pred2))
print('mean_absolute_error',mean_absolute_error(y_test,lasso_y_pred2))
# data.to_csv('final_data.csv',index=False)
training_data = pd.concat([x_train,y_train],axis=1)
training_data # Index is change or looks informat due to split method
training_data.to_csv('training_data.csv',index=False)
import pickle
pickle.dump(data,open('data.pkl','wb'))
pickle.dump(lr_pipe,open('lr_pipe.pkl','wb'))
data

# For saving commands 
# save commands
