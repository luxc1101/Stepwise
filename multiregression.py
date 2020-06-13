import pandas as pd
import numpy as np
import stepwiseregrssion
import stepwiseregrssion_F_P
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy import stats
import figsave
import os

PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)
'''
Boston house prices dataset
---------------------------
**Data Set Characteristics:**  
    :Number of Instances: 506 
    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
    :Missing Attribute Values: None
    :Creator: Harrison, D. and Rubinfeld, D.L.
This is a copy of UCI ML housing dataset.
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

p-value:
https://de.wikipedia.org/wiki/P-Wert
Ist der p-Wert „klein“ (kleiner als ein vorgegebenes Signifikanzniveau; allgemein < 0,05), so lässt sich die Nullhypothese ablehnen.
Anders ausgedrückt: Ist die errechnete Prüfgröße größer als der kritische Wert (kann unmittelbar aus einer Quantiltabelle abgelesen werden), 
so kann die Nullhypothese verworfen werden und man kann davon ausgehen, 
dass die Alternativhypothese gilt und damit ein bestimmter Zusammenhang besteht (z.B. ein neues Medikament ist wirksam). 
Wenn die Nullhypothese zugunsten der Alternativhypothese verworfen wird, wird das Resultat als „statistisch signifikant“ bezeichnet.
festgesetzte Grenzen wie 5%, 1% oder 0.1%
'''
my_seed = 20
test_size = 0.05
threshold_in = 0.05
threshold_out = 0.1

data = load_boston()
X_all_data = pd.DataFrame(data.data, columns=data.feature_names)
y_all_targe = data.target
all_data = pd.concat([X_all_data,pd.Series(y_all_targe,name = 'MEDV')],axis=1)
'''
------------------------------------------------------------
**Correlation analysis between extracted features and MEDV**
------------------------------------------------------------
'''
def corr_ranking(data, target_col):
    corr_matrix = all_data.corr()
    index = list(corr_matrix.index)
    index.remove(target_col)
    index_ranking = abs(corr_matrix.loc[index,target_col]).sort_values(ascending = False)
    val_ranking = sorted(corr_matrix.loc[index,target_col],key = abs,reverse = True)
    corr_ranking = pd.DataFrame(val_ranking,columns = [target_col],index = index_ranking.index)
    return corr_ranking

corr_ranking(all_data,'MEDV').plot.bar()
plt.title('Korrelationskoeffizient zum Target')
plt.grid(axis = 'y')
figsave.save_fig(IMAGES_PATH,'Korrelationskoeffizient zum Target')
plt.show()

# # split the data to traindata and test data 
X_train, X_test, y_train, y_test = train_test_split(X_all_data, y_all_targe, test_size=test_size,random_state=my_seed)
# data normalisation
scaler = StandardScaler()
X_train_nor = pd.DataFrame(scaler.fit_transform(X_train.values), index=X_train.index, columns=X_train.columns) 
X_test_nor = pd.DataFrame(scaler.transform(X_test.values), index=X_test.index, columns=X_test.columns)
'''
-----------------------
**Stepwiseregression**
-----------------------
''' 
result_model, subfeature, final_formula = stepwiseregrssion_F_P.stepwise_fvalue(X_train_nor,y_train,threshold_in,threshold_out)
print(result_model.summary())
print('final formula:',final_formula)
# predict
y_pred = result_model.predict(sm.add_constant(X_test_nor)[subfeature]).to_numpy()
# MSE & MSE
reg_mse = mean_squared_error(y_test,y_pred)
reg_mae = mean_absolute_error(y_test,y_pred)
print('MSE:',reg_mse)
print('MAE:',reg_mae)
# plot
fig, ax = plt.subplots(1,3,figsize =(20,5))
fig.suptitle('Stepwiseregression zur Prognose vom Hauspreis im Boston') 

ax[0].plot(np.arange(0,len(y_test)*2,2),y_test,'o',label = 'y true')
ax[0].plot(np.arange(1,len(y_pred)*2+1,2),y_pred,'*',color='gray',label = 'y pred')
for x,y in zip(np.arange(0,len(y_test)*2,2),y_test):
	ax[0].plot([x,x],[0,y],'--',color ='#1f77b4',lw=0.5)
for x,y in zip(np.arange(1,len(y_pred)*2+1,2),y_pred):
	ax[0].plot([x,x],[0,y],'--',color='gray',lw=0.5)
ax[0].set_title(final_formula,fontsize = 8)
ax[0].set_ylabel('Median value in $1000')
ax[0].set_xlim(0,len(y_test)*2)
ax[0].set_ylim(0)
ax[0].set_xticks([])
ax[0].legend()
ax[0].text(1,40,'MSE: {:.4}\nMAE: {:.4} '.format(reg_mse,reg_mae),horizontalalignment='left',verticalalignment='center',fontsize=8)
# Residual
ax[1].scatter(np.arange(0,len(y_test)),y_test-y_pred,marker ='x',label = 'y true - y pred')
ax[1].plot([0,len(y_test)],[0,0],'k--')
ax[1].set_ylabel('Median value in $1000')
ax[1].legend()
ax[1].set_title('Residual Plots',fontsize = 10)
# Probplot
stats.probplot(y_test-y_pred,fit =True,plot=ax[2])
figsave.save_fig(IMAGES_PATH,'Stepwiseregression')
plt.show()



