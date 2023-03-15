import matplotlib as mpl
import matplotlib.pyplot as mp
import numpy as np
params = {
	'font.family': 'serif',
	'text.usetex': True,
	'pgf.rcfonts': False,
	'pgf.texsystem': 'xelatex',
	'pgf.preamble': r'\usepackage{fontspec,physics}',
}

mpl.rcParams.update(params)
fig = mp.figure(figsize=(6.0, 2.5))
fig.subplots_adjust(left=0.12, top=.7, right=0.999, bottom=0.248)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)
ax.xaxis.grid(linestyle=':', color='grey', linewidth=0.5)

x = np.load('vars.npy')



ax.plot(x[2][0], x[2][1], label='Fixed Noise (' + r'$\frac{\pi}{8}$' + ')', color='#76D7C4')
ax.plot(x[0][0], x[0][1], label='Fixed Noise (' + r'$\frac{\pi}{2}$' + ')', color='#CB4335',linestyle=':')
ax.plot(x[1][0], x[1][1], label='Adaptive Noise', color='#000000')
ax.set_xlim(0, 5.9)
ax.set_ylim(0, 1)
#ax.set_xticks(range(10))
#ax.set_xticklabels([i for i in range(10)], rotation=0, ha='right')
ax.set_xlabel('Variation Across Images', fontsize=14)
mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)
ax.set_ylabel('CDF', fontsize=14)
ax.legend(ncol=3, edgecolor='black', bbox_to_anchor=(0, 1.02, 1., .102),
            loc='lower left', borderaxespad=0.02,
            fontsize=14, handletextpad=0.2, handlelength=1.6,borderpad=0.29,columnspacing=0.5)
mp.savefig('cdf.pdf',bbox_inches='tight',pad_inches=0.01)
mp.show()
mp.close()
