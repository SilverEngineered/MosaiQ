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


fids = np.load('fids_dist5Again.npy')
fids2 = np.load('fids_dist5noorder2.npy')

# count, bins = np.histogram(fids, bins=20)
# pdf = count / sum(count)
# cdf = np.cumsum(pdf)
# x = np.sort(cdf)
# y = np.arange(20) / float(20)
# print(x)
# print(y)
count, bins = np.histogram(fids, bins=15)
bins = np.linspace(40, 46, 50)

# count, bins = np.histogram(fids2, bins=20)
# pdf = count / sum(count)
# cdf = np.cumsum(pdf)
# x = np.sort(cdf)
# y = np.arang#e(20) / float(20)
#ax.plot(x, y, label='Without Remapping', color='#52B788')

ax.hist(fids2, bins, label='MosaiQ without PCA Feature Redistribution', color='#000000', linewidth=1.0,
    hatch='////',edgecolor='white')
ax.hist(fids, bins, label='MosaiQ with PCA Feature Redistribution', color='#000000', linewidth=1.0,
        edgecolor='black')
ax.set_xlim(40.5, 45.65)
ax.set_ylim(0, 30)
#ax.set_xticks(range(10))
#ax.set_xticklabels([i for i in range(10)], rotation=0, ha='right')
ax.set_xlabel('FID Score', fontsize=14)
mp.setp(ax.get_xticklabels(), fontsize=14)
ax.set_yticklabels(['0', '0.05', '0.10', '0.15'])
mp.setp(ax.get_yticklabels(), fontsize=14)
ax.set_ylabel('PDF', fontsize=14)
ax.legend(ncol=1,edgecolor='black', bbox_to_anchor=(0, 1.02, 1., .102),
            loc='lower left', borderaxespad=0.02,
            fontsize=14, handletextpad=0.5, mode='expand')
mp.savefig('cdf_fid.pdf', pad_inches=0.01, bbox_inches='tight')
mp.close()
