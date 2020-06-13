import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

my_seed = 20
test_size = 0.05
threshold_in = 0.05
threshold_out = 0.1

data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=my_seed)
# data normalisation
scaler = StandardScaler()
X_train_nor = pd.DataFrame(scaler.fit_transform(X_train.values), index=X_train.index, columns=X_train.columns) 
X_test_nor = pd.DataFrame(scaler.transform(X_test.values), index=X_test.index, columns=X_test.columns)


def stepwise_fvalue(X,y,  
			threshold_in = 0.01,
			threshold_out = 0.05,
			verbose = True
			):

	def regressior(y,X):
		 regressor = sm.OLS(y, X).fit()
		 return regressor

	X = sm.add_constant(X)
	selected_cols = ["const"]
	other_cols = list(X.columns).copy()
	other_cols.remove('const')

	# criteria on beginn
	model = regressior(y,X[selected_cols])
	criteria = abs(model.fvalue)
	print(criteria)

	for i in range(X.shape[1]):
		f_with_variate=[]
		fpvals = pd.DataFrame(columns = ['Reduced_Mode','F_st','Pval'])
		for j in other_cols:
			model = regressior(y,X[selected_cols+[j]])
			fpvals = fpvals.append(pd.DataFrame([[selected_cols+[j],model.fvalue,model.pvalues[j]]],columns = ["Reduced_Mode",'F_st',"Pval"]),ignore_index=True)
		print(fpvals)
		fpvals = fpvals[fpvals.Pval<=threshold_in].reset_index(drop=True)
		if fpvals.shape[0]>0:
			new_add_l = [fpvals.Reduced_Mode[k][-1] for k in range(len(fpvals))]
			for col in new_add_l:
				model = regressior(y,X[selected_cols + [col]])
				f_with_variate.append((model.fvalue, col))
				f_with_variate.sort()
			best_f, best_candidate = f_with_variate.pop()
			selected_cols.append(best_candidate)
			other_cols.remove(best_candidate)
			if verbose:
				print("Entered : {:15} F-value :{:.6}".format(best_candidate,best_f))
				print('Selected Col : {}\n'.format(selected_cols))

		else:
			print("Break : Significance Level")
			break
	model = regressior(y,X[selected_cols])
	for x,y in zip(model.params.index, model.params.values):
		if x =='const':
			final_formula='{:.3}'.format(y)
		else:
			final_formula = final_formula + ('-{:.3}{}'.format(abs(y),x) if y<0 else '+{:.3}{}'.format(y,x))
	final_formula = 'Y = '+final_formula

	return model, selected_cols, final_formula


result_model, subfeature, final_formula = stepwise_fvalue(X_train_nor,y_train,threshold_in)
print(result_model.summary())
print(final_formula)

y_pred = result_model.predict(sm.add_constant(X_test_nor)[subfeature]).to_numpy()
# MSE & MSE
reg_mse = mean_squared_error(y_test,y_pred)
reg_mae = mean_absolute_error(y_test,y_pred)
print('MSE:',reg_mse)
print('MAE:',reg_mae)
