import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker
from matplotlib.ticker import FormatStrFormatter
from IPython import get_ipython



def reset_plots():
	"""
	Makes axes large and enables LaTeX for matplotlib plots
	"""
	plt.close('all')
	fontsize = 20
	legsize = 15
	plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
	plt.rc('text', usetex = True)
	font = {'size': fontsize}
	plt.rc('font', **font)
	rc = {'axes.labelsize': fontsize,
		  'font.size': fontsize,
		  'axes.titlesize': fontsize,
		  'xtick.labelsize': fontsize,
		  'ytick.labelsize': fontsize,
		  'legend.fontsize': legsize}

	mpl.rcParams.update(**rc)
	mpl.rc('lines', markersize=10)
	plt.rcParams.update({'axes.labelsize': fontsize})
	mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
										   r'\usepackage{amsfonts}']






def plot(nrows=1, ncols=1, fig_size=5):
	"""
	Generate a matplotlib plot and axis handle

	Parameters
	--------------
	nrows: an int, number of rows for subplotting
	ncols: an int, number of columns for subplotting
	fig_size: numeric or array (xfigsize, yfigsize) - the size of each axis.

	NOTE: returns the axis as a flattened array
	"""

	if isinstance(fig_size, (list, tuple)):
		xfigsize, yfigsize = fig_size
	elif isinstance(fig_size, (int, float)):
		xfigsize = yfigsize = fig_size
	else:
		raise ValueError('Invalid fig_size type')

	fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*xfigsize, nrows*yfigsize))
	if nrows*ncols > 1:
		ax = ax.ravel()
	return fig, ax




def remove_tex_axis(ax, xtick_fmt='%d', ytick_fmt='%d', axis_remove='both'):
	"""
	Makes axes normal font in matplotlib

	Parameters
	-------------
	xtick_fmt: string, defining the format of the x-axis
	ytick_fmt: string, defining the format of the y-axis
	axis_remove: string, which axis to remove ['x', 'y', 'both']
	"""
	if axis_remove not in ['x', 'y', 'both']:
		raise Exception('axis_remove value not allowed')
	fmt = matplotlib.ticker.strMethodFormatter("{x}")

	if (axis_remove == 'both'):
		ax.xaxis.set_major_formatter(fmt)
		ax.yaxis.set_major_formatter(fmt)
		ax.xaxis.set_major_formatter(FormatStrFormatter(xtick_fmt))
		ax.yaxis.set_major_formatter(FormatStrFormatter(ytick_fmt))
	elif (axis_remove == 'x'):
		ax.xaxis.set_major_formatter(fmt)
		ax.xaxis.set_major_formatter(FormatStrFormatter(xtick_fmt))
	else:
		ax.yaxis.set_major_formatter(fmt)
		ax.yaxis.set_major_formatter(FormatStrFormatter(ytick_fmt))




def simpleaxis(ax):
	"""
	Remove top and right spines from a plot

	Parameters
	-------------
	ax: matplotlib axis
	"""
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()



def legend_outsize(ax, pointers=None, labels=None, size=15, frameon=True):
	"""
	Put legend outside the plot area

	Parameters
	-------------
	ax: matplotlib axis
	"""

	if (pointers is None) and (labels is None):
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
					prop={'size': size}, frameon=frameon)
	else:
		assert len(pointers) == len(labels)
		ax.legend(pointers, labels, loc='center left', bbox_to_anchor=(1, 0.5),
					prop={'size': size}, frameon=frameon)



def to_latex(x, dp=1, double_backlash=True):
	"""
	Convert a decimal into LaTeX scientific notation

	Parameters
	--------------
	x: float, the number to convert to LaTeX notation, e. g. 0.66
	dp: int, the number of decimal places for the
	double_backslash: bool, whether to use a double-backslash for the LaTeX commands

	Returns
	----------
	String, where x is cast in LaTeX scientific notation, e. g. "6.6 \times 10^{-1}"

	"""
	fmt = "%.{}e".format(dp)
	s = fmt % x
	arr = s.split('e')
	m = arr[0]
	n = str(int(arr[1]))
	if double_backslash:
		return str(m) + '\\times 10^{' + n + '}'
	else:
		return str(m) + '\times 10^{' + n + '}'

	
