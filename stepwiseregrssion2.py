import numpy as np
from  sklearn import datasets
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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