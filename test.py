import numpy as np
from  sklearn import datasets
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
#---------------------------------------------------------------------------------------------------------------------------------------#
# test_size = 0.2
# my_seed = 3

# data = datasets.load_boston()
# X_all_data = pd.DataFrame(data.data, columns=data.feature_names)
# y_all_targe = data.target
# all_data = pd.concat([X_all_data,pd.Series(y_all_targe,name = 'MEDV')],axis=1)

# def corr_ranking(data, target_col):
#     corr_matrix = all_data.corr()
#     index = list(corr_matrix.index)
#     index.remove(target_col)
#     index_ranking = abs(corr_matrix.loc[index,target_col]).sort_values(ascending = False)
#     val_ranking = sorted(corr_matrix.loc[index,target_col],key = abs,reverse = True)
#     corr_ranking = pd.DataFrame(val_ranking,columns = [target_col],index = index_ranking.index)
#     return corr_ranking

# # # split the data to traindata and test data 
# X_train, X_test, y_train, y_test = train_test_split(X_all_data, y_all_targe, test_size=test_size,random_state=my_seed)
# # data normalisation
# scaler = StandardScaler()
# X_train_nor = pd.DataFrame(scaler.fit_transform(X_train.values), index=X_train.index, columns=X_train.columns) 
# X_test_nor = pd.DataFrame(scaler.transform(X_test.values), index=X_test.index, columns=X_test.columns)
# print('Bearbietung der Daten Done')
# ---------------------------------------------------------------------------------------------------------------------------------------#
my_seed = 3
test_size = 0.3

iris = datasets.load_iris()
X_all_data = pd.DataFrame(iris.data, columns=iris.feature_names)
columns_l = iris.feature_names
X_all_data.rename(columns = {columns_l[0]:'sepal_l',columns_l[1]:'sepal_w',columns_l[2]:'petal_l',columns_l[3]:'petal_w'},inplace = True)
y_all_targe = iris.target
all_data = pd.concat([X_all_data,pd.Series(y_all_targe,name = 'class')],axis=1)
# # split the data to traindata and test data 
X_train, X_test, y_train, y_test = train_test_split(X_all_data, y_all_targe, test_size=test_size,random_state=my_seed)
# data normalisation
scaler = StandardScaler()
X_train_nor = pd.DataFrame(scaler.fit_transform(X_train.values), index=X_train.index, columns=X_train.columns) 
X_test_nor = pd.DataFrame(scaler.transform(X_test.values), index=X_test.index, columns=X_test.columns)
print('Bearbietung der Daten Done')
#---------------------------------------------------------------------------------------------------------------------------------------#
# Daten Vorbereitung Done
#---------------------------------------------------------------------------------------------------------------------------------------#
# threshold_in = 0.05
# threshold_out = 0.1
# verbose = True

def stepwise_reg(X,y,
			model_type = 'linear',
			elimination_criteria = 'pvalue',
			threshold_in = 0.01,
			threshold_out = 0.05,
			verbose = True,
			my_seed = 3
			):
	""" a forward-backward feature selection
	beased on F statistic in combination with the p value from statsmondels.api.OLS
	Argeuments:
		X - pandas.DataFrame with candidate feature
		y - target
		model_type - 'linear' or 'MNlogit'
		elimination_criteria - 'pvalue' or 'aic'
		threshold_in - include a feature if its p < threshold_in
		threshold_out - exclude a freature if its p > threshold_out
		verbose - weather to print the sequence of inclusion and exclusions
	return:
		Model and list of selected features
	Always set threshold_in < threshold_out.
	"""
	# def regressor(y,X):
	# 	 regressor = sm.OLS(y, X).fit()
	# 	 return regressor

	def regressor(y,X, model_type=model_type):
		if model_type == "linear":
		    regressor = sm.OLS(y, X).fit()
		elif model_type == "MNlogit":
		    regressor = sm.MNLogit(y, X).fit(method='lbfgs',maxiter = 100,disp = 0)
		else:
		    print("\nWrong Model Type : "+ model_type +"\nLinear model type is seleted.")
		    model_type = "linear"
		    regressor = sm.OLS(y, X).fit()
		return regressor

	X = sm.add_constant(X)
	selected_cols = ["const"]
	other_cols = list(X.columns).copy()
	other_cols.remove('const')

	if elimination_criteria == 'aic' and model_type =='MNlogit':
		model = regressor(y, X[selected_cols])
		criteria = model.aic
		for i in range(X.shape[1]):
			changed = False
			aic = pd.DataFrame(columns = ['Cols','aic'])
			for j in other_cols:
				model = regressor(y,X[selected_cols + [j]])
				aic = aic.append(pd.DataFrame([[j, model.aic]],columns = ["Cols","aic"]),ignore_index=True)
			aic = aic.sort_values(by = ["aic"]).reset_index(drop=True)
			print('\n-------------forward step-------------\n')
			print(aic)
			model = regressor(y, X[selected_cols+[aic["Cols"][0]]])
			new_criteria = model.aic
			if new_criteria < criteria:
				selected_cols.append(aic["Cols"][0])
				other_cols.remove(aic["Cols"][0])
				criteria = new_criteria
				changed = True
				if verbose:
					print("Entered : {:10} aic :{:.6}".format(aic["Cols"][0],model.aic))
			if not changed:
				print("Break : Criteria")
				break
		model = regressor(y,X[selected_cols])
		return model,selected_cols

	elif elimination_criteria == 'pvalue' and model_type == 'linear':
		for i in range(X.shape[1]):
			changed = False
			f_with_variate=[]
			pvals = pd.DataFrame(columns = ['Cols','Pval'])
			for j in other_cols:
				model = regressor(y,X[selected_cols+[j]])
				pvals = pvals.append(pd.DataFrame([[j, model.pvalues[j]]],columns = ["Cols","Pval"]),ignore_index=True)
			print('\n-------------forward step-------------\n')
			print(pvals)
			# only Pvalues litter than threshold in will be accepted
			pvals = pvals[pvals.Pval<=threshold_in]
			if pvals.shape[0]>0:
				for col in pvals.Cols:
					model = regressor(y,X[selected_cols + [col]])
					f_with_variate.append((model.fvalue,col))
				f_with_variate.sort()
				max_fvalue, best_candidate = f_with_variate.pop()
				selected_cols.append(best_candidate)
				
			# after add will all condidate be checked to make sure all pvalue litter than threshold out
				print('\n-------------backward step-------------\n')
				model = regressor(y,X[selected_cols])
				pvalues = model.pvalues.iloc[1:]
				worst_pval = pvalues.max()
				if worst_pval > threshold_out:
					worst_feature = pvalues.idxmax()
					selected_cols.remove(worst_feature)
					changed = True
					if verbose:
						print('Drop {:10}with p-value {:.6}'.format(worst_feature,worst_pval))
				else:
					other_cols.remove(best_candidate)
					changed = True
					if verbose:
						print("Entered : {:10} F-value :{:.6}".format(best_candidate,max_fvalue))
			
			if not changed:
				print("Break : Significance Level")
				break

		model = regressor(y,X[selected_cols])		
		# final formula
		for x,y in zip(model.params.index, model.params.values):
			if x =='const':
				final_formula='{:.3}'.format(y)
			else:
				final_formula = final_formula + ('-{:.3}{}'.format(abs(y),x) if y<0 else '+{:.3}{}'.format(y,x))
		final_formula = 'Y = '+final_formula

		return model, selected_cols, final_formula


a = stepwise_reg(X_train_nor,y_train,model_type = 'MNlogit',elimination_criteria='aic')
print(a)
y_pred = a[0].predict(sm.add_constant(X_test_nor)[a[1]]).idxmax(axis= 1)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(a[0].summary())
# print(final_formula)


#---------------------------------------------------------------------------------------------------------------------------------------#
# Sequential Feature Selector
#---------------------------------------------------------------------------------------------------------------------------------------#
# from  sklearn.linear_model import  LinearRegression
# from sklearn.neighbors import KNeighborsRegressor
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# import matplotlib.pyplot as plt 
# from sklearn.metrics import mean_absolute_error
# pd.set_option('display.max_columns', None)

# knn= KNeighborsRegressor(n_neighbors=10)
# reg = LinearRegression()
# X = X_train_nor
# y = y_train
# sffs = SFS(reg, 
#            k_features=None, 
#            forward=True, 
#            floating =True,
#            scoring='r2',
#            cv=5)
# pipe = Pipeline([('sffs', sffs), ('reg', reg)])
# print(pipe)
# param_grid = [{'sffs__k_features': [8,9,10,11,12,13]}]

# gs = GridSearchCV(estimator=pipe, 
#                   param_grid=param_grid, 
#                   scoring='r2', 
#                   n_jobs=-1, 
#                   cv=5)

# gs = gs.fit(X, y)

# print(gs.best_params_)
# print(gs.best_estimator_.steps[0][1].k_feature_names_)
# best_model = gs.best_estimator_
# print(best_model[1].coef_)
# y_pred = best_model.predict(X_test_nor)

# mae = mean_absolute_error(y_test, y_pred)
# print("\nMAE: \t" + str("%.5f" % mae))

# min_test, max_test = y_test.min(), y_test.max()
# plt.figure(figsize=(8, 5))
# plt.ylabel('vorhergesagte price')
# plt.xlabel('reale price')
# plt.title("Boston house price")
# plt.scatter(y_test, y_pred, alpha=0.5, s=100, edgecolors="k")
# plt.plot([min_test, max_test], [min_test, max_test], "k--", linewidth=2)
# plt.show()