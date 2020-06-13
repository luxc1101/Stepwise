import numpy as np
import statsmodels.api as sm
import pandas as pd

def stepwise_pvalue(X,y, 
			initial_list = [], 
			threshold_in = 0.001,
			threshold_out = 0.01,
			verbose = True
			):
	""" a forward-backward feature selection
	beased on p-value from statsmondels.api.OLS
	Argeuments:
	
		X - pandas.DataFrame with candidate feature
		y - target
		initial_list - list of feature to start with (cols names of X)
		threshold_in - include a feature if its p < threshold_in
		threshold_out - exclude a freature if its p > threshold_out
		verbose - weather to print the sequence of inclusion and exclusions
	return:
		Model and list of selected features
	Always set threshold_in < threshold_out.
	"""

	included = list(initial_list)

	if threshold_in <= threshold_out:
		while True:
			changed = False
			# forward selection
			excluded = list(set(X.columns) - set(included))
			new_pval = pd.Series(index = excluded)
			for new_col in excluded:
				model = sm.OLS(y, sm.add_constant(X[included+[new_col]])).fit()
				new_pval[new_col] = model.pvalues[new_col]	
			best_pval = new_pval.min()
			if best_pval < threshold_in:
				best_feature = new_pval.idxmin()
				included.append(best_feature)
				changed = True
				if verbose:
					print('Add {:30}with p-value {:.6}'.format(best_feature,best_pval))
			# backward selction
			model = sm.OLS(y,sm.add_constant(X[included])).fit()
			pvalues = model.pvalues.iloc[1:]
			worst_pval = pvalues.max()
			if worst_pval > threshold_out:
				worst_feature = pvalues.idxmax()
				included.remove(worst_feature)
				changed = True
				if verbose:
					print('Drop {:30}with p-value {:.6}'.format(worst_feature,worst_pval))

			if not changed:
				break
		# final formula
		for x,y in zip(model.params.index, model.params.values):
			if x =='const':
				final_formula='{:.3}'.format(y)
			else:
				final_formula = final_formula + ('-{:.3}{}'.format(abs(y),x) if y<0 else '+{:.3}{}'.format(y,x))
		final_formula = 'y = '+final_formula
		
		return model,included,final_formula
	
	else:
		return  (print('Warning: threshold_in sollte kleiner als threshold_out'))


