import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def stepwise_fvalue(X,y,  
			threshold_in = 0.01,
			threshold_out = 0.05,
			verbose = True
			):
	""" a forward-backward feature selection
	beased on F statistic in combination with the p value from statsmondels.api.OLS
	Argeuments:
		X - pandas.DataFrame with candidate feature
		y - target
		threshold_in - include a feature if its p < threshold_in
		threshold_out - exclude a freature if its p > threshold_out
		verbose - weather to print the sequence of inclusion and exclusions
	return:
		Model and list of selected features
	Always set threshold_in < threshold_out.
	"""
	def regressior(y,X):
		 regressor = sm.OLS(y, X).fit()
		 return regressor

	X = sm.add_constant(X)
	selected_cols = ["const"]
	other_cols = list(X.columns).copy()
	other_cols.remove('const')

	for i in range(X.shape[1]):
		changed = False
		f_with_variate=[]
		pvals = pd.DataFrame(columns = ['Cols','Pval'])
		for j in other_cols:
			model = regressior(y,X[selected_cols+[j]])
			pvals = pvals.append(pd.DataFrame([[j, model.pvalues[j]]],columns = ["Cols","Pval"]),ignore_index=True)
		print('\n-------------forward step-------------\n')
		print(pvals)
		# only Pvalues litter than threshold in will be accepted
		pvals = pvals[pvals.Pval<=threshold_in]
		if pvals.shape[0]>0:
			for col in pvals.Cols:
				model = regressior(y,X[selected_cols + [col]])
				f_with_variate.append((model.fvalue,col))
			f_with_variate.sort()
			max_fvalue, best_candidate = f_with_variate.pop()
			selected_cols.append(best_candidate)
			
		# after add will all condidate be checked to make sure all pvalue litter than threshold out
			print('\n-------------backward step-------------\n')
			model = regressior(y,X[selected_cols])
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

	model = regressior(y,X[selected_cols])		
	# final formula
	for x,y in zip(model.params.index, model.params.values):
		if x =='const':
			final_formula='{:.3}'.format(y)
		else:
			final_formula = final_formula + ('-{:.3}{}'.format(abs(y),x) if y<0 else '+{:.3}{}'.format(y,x))
	final_formula = 'Y = '+final_formula

	return model, selected_cols, final_formula




