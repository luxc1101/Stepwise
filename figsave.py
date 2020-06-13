import os
import matplotlib.pyplot as plt

def save_fig(image_path,
		fig_id,
		bbox_inches ='tight', 
		fig_extension ='png', 
		reselution = 300,
		verbose = True
		):
	""" a function to save fig
		image_path - path
		fig_id - name of img
		tight_layout - automatically adjust subplot parameters to give specified padding
		fig_extension - default 'png'
		reselution - default 300
		verbose - print fig_id
	"""
	path = os.path.join(image_path,fig_id + '.' + fig_extension)
	if verbose:
		print('Saving figture', fig_id)
	# if tight_layout:
	# 	plt.tight_layout()
	plt.savefig(path,dpi=reselution,bbox_inches = bbox_inches)