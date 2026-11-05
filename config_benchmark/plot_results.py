"""Render documentation figures from recorded measurements; never train a model."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, default=Path(__file__).with_name('creative_results.json'))
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('figures'))
args = parser.parse_args()
r = json.loads(args.results.read_text())
args.output.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
files = []
def save(fig, name):
    fig.savefig(args.output / (name + '.png'), dpi=160, bbox_inches='tight')
    fig.savefig(args.output / (name + '.svg'), bbox_inches='tight')
    files.extend([name + '.png', name + '.svg'])
    plt.close(fig)

runs = sorted([x for x in r['runs'] if x['stage'] == 'screen'], key=lambda x:x['validation']['macro_f1'])
fig, ax = plt.subplots(figsize=(10, 6))
names = [' → '.join(map(str, r['configs'][x['name']]['layers'][1:-1])) +
         (' (init 0.2)' if r['configs'][x['name']]['init'] == .2 else '') for x in runs]
values = [100*x['validation']['macro_f1'] for x in runs]
ax.barh(names, values, color=['#e59a37' if x['name']=='deep64' else '#217c88' for x in runs])
for i,v in enumerate(values): ax.text(v+.3,i,f'{v:.2f} %',va='center',fontsize=9)
ax.set(xlim=(0,65), xlabel='F1 macro de validation (%)', title='11 architectures — présélection à graine 42')
ax.text(0,-.16,'3 000 entraînement · 2 000 validation · jusqu’à 40 époques · checkpoint au minimum de perte',transform=ax.transAxes,fontsize=9)
save(fig,'architectures')

fig, axes = plt.subplots(1,2,figsize=(13,4.8))
colors={'baseline256':'#217c88','deep64':'#d7801c'}
for x in r['runs']:
    if x['stage']!='final': continue
    h=x['history']; color=colors[x['name']]; label=f"{'256' if x['name']=='baseline256' else '64 × 4'} / graine {x['seed']}"
    for ax,key in zip(axes,['loss','macro_f1']):
        ax.plot([v['epoch'] for v in h],[v['validation'][key] for v in h],
                color=color,linestyle={42:'-',43:'--',44:':'}[x['seed']],label=label)
        best=next((v for v in h if v['epoch']==x['best_epoch']),None)
        if best: ax.scatter([best['epoch']],[best['validation'][key]],color=color,s=30,zorder=5)
        ax.set_xlabel('Époque'); ax.grid(alpha=.2)
axes[0].set_ylabel('Perte quadratique moyenne'); axes[1].set_ylabel('F1 macro')
axes[0].legend(fontsize=8);fig.suptitle('Validation des finalistes — points : checkpoint restauré (minimum de perte)')
fig.tight_layout();save(fig,'learning_curves')

fig, ax = plt.subplots(figsize=(7,5.7))
m = r['test']['42']; matrix=np.asarray(m['confusion'])
im=ax.imshow(matrix,cmap='Blues',vmin=0)
labels=r['configs'][r['winner']]['labels']
ax.set_xticks(range(5),labels,rotation=35,ha='right');ax.set_yticks(range(5),labels)
for i in range(5):
 for j in range(5): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()*.6 else 'black')
ax.set(xlabel='Classe prédite',ylabel='Classe attendue',title='Test : réseau 256, graine 42 — 6 000 positions')
fig.colorbar(im,ax=ax,label='Nombre de positions',shrink=.8);save(fig,'confusion')

fig, axes = plt.subplots(1,2,figsize=(10,4.5))
for name,color in colors.items():
 group=[x for x in r['runs'] if x['stage']=='final' and x['name']==name]
 for ax,key in zip(axes,['seconds','parameters']):
  ax.scatter([x[key] for x in group],[100*x['validation']['macro_f1'] for x in group],color=color,label='256' if name=='baseline256' else '64 × 4',s=65)
for ax in axes:
 ax.set_ylabel('F1 macro validation (%)');ax.grid(alpha=.2);ax.legend()
axes[0].set_xlabel('Temps des époques + validation (s)');axes[1].set_xlabel('Nombre de paramètres')
fig.suptitle('Compromis qualité / coût — un point par graine, 6 000 positions d’entraînement')
fig.tight_layout();save(fig,'tradeoffs')
manifest={'source':args.results.name,'sha256':hashlib.sha256(args.results.read_bytes()).hexdigest(),
          'matplotlib':matplotlib.__version__,'files':files,
          'note':'Figures from stored results, no new training; test confusion uses seed 42 only.'}
(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Generated',len(files),'figures')
