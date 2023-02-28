import matplotlib as mpl
import matplotlib.pyplot as mp

params = {
	'font.family': 'serif',
	'text.usetex': True,
	'pgf.rcfonts': False,
	'pgf.texsystem': 'xelatex',
	'pgf.preamble': r'\usepackage{fontspec,physics}',
}

mpl.rcParams.update(params)
fig = mp.figure(figsize=(12.0, 3.0))
fig.subplots_adjust(left=0.068, top=0.86, right=0.999, bottom=0.245)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=':', color='grey', linewidth=0.5)
values = [1,2,3,4,5,6,7,8,9,0]
ax.bar([i - 0.2 for i in range(10)],
        values,
        color='#000000',
        label='Uncorrected',
        linewidth=1.0,
        edgecolor='black',
        width=0.4)
values = [0,1,2,3,4,5,6,7,8,9]
ax.bar([i + 0.2 for i in range(10)],
        values,
        color='#276927',
        label='Corrected (' + r'\textsc{MosaiQ}' + ')',
        linewidth=1.0,
        edgecolor='black',
        width=0.4)
ax.set_xlim(-0.6, 10-0.4)
ax.set_ylim(0, 100)
ax.set_xticks(range(10))
ax.set_xticklabels([i for i in range(10)], rotation=30, ha='right')
mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)
ax.set_ylabel('Fréchet inception distance (FID)\n' + r'\textit{(lower is better)}', fontsize=14)
ax.legend(ncol=2, edgecolor='black', bbox_to_anchor=(0.6365, 1.02, 1., .102),
            loc='lower left', borderaxespad=0.02,
            fontsize=14, handletextpad=0.5)
mp.savefig('fid.pdf')
mp.close()