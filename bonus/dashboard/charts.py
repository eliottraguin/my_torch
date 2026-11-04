"""Interactive experiment charts and self-contained HTML exports."""

import csv
from io import StringIO

import plotly.graph_objects as go

COLORS = ['#52dacb', '#ffbd69', '#a896ff', '#ff819c', '#7cb8ff', '#d4e87e']
BG = '#111b2c'


def style(figure, title='', height=340):
    figure.update_layout(title=title, height=height, paper_bgcolor=BG, plot_bgcolor=BG,
                         font=dict(color='#dae4f2', family='Inter, sans-serif', size=12),
                         margin=dict(l=25, r=25, t=55, b=35),
                         legend=dict(orientation='h', y=1.15),
                         colorway=COLORS, hovermode='x unified')
    figure.update_xaxes(gridcolor='#253047', zeroline=False, automargin=True)
    figure.update_yaxes(gridcolor='#253047', zeroline=False, automargin=True)
    return figure


def epoch_records(history):
    return [row for row in history if row.get('event') == 'epoch']


def learning_chart(history, metric='loss', logarithmic=False):
    rows = epoch_records(history)
    figure = go.Figure()
    for key, label, color in [('train', 'Entraînement', COLORS[0]),
                              ('validation', 'Validation', COLORS[1])]:
        values = [r[key][metric] if r.get(key) else None for r in rows]
        if any(value is not None for value in values):
            figure.add_trace(go.Scatter(x=[r['epoch'] for r in rows], y=values,
                                        name=label, mode='lines+markers',
                                        line=dict(color=color, width=3), marker=dict(size=5)))
    figure.update_xaxes(title='Époque')
    if metric == 'accuracy':
        figure.update_yaxes(title='Bonnes prédictions', tickformat='.0%', range=[0, 1])
    else:
        figure.update_yaxes(title='Perte quadratique moyenne', type='log' if logarithmic else 'linear')
    return style(figure, 'Précision à poids fixes' if metric == 'accuracy' else 'Courbes d’apprentissage')


def confusion_chart(matrix, labels, normalize=False):
    import numpy as np
    values = np.asarray(matrix, dtype=float)
    if normalize:
        sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, sums, out=np.zeros_like(values), where=sums > 0)
    figure = go.Figure(go.Heatmap(z=values, x=labels, y=labels,
                                  colorscale=[[0, '#18243a'], [1, COLORS[0]]],
                                  texttemplate='%{z:.0%}' if normalize else '%{z:.0f}',
                                  hovertemplate='Attendu : %{y}<br>Prédit : %{x}<br>Valeur : %{z}<extra></extra>'))
    figure.update_xaxes(title='Classe prédite', side='bottom')
    figure.update_yaxes(title='Classe attendue', autorange='reversed')
    return style(figure, 'Matrice de confusion', height=440)


def distribution_chart(counts, title):
    figure = go.Figure(go.Bar(x=list(counts), y=list(counts.values()),
                              marker_color=COLORS[0], text=list(counts.values()), textposition='auto'))
    return style(figure, title)


def per_class_chart(rows):
    figure = go.Figure()
    for metric, name, color in [('precision', 'Précision', COLORS[0]),
                                 ('recall', 'Rappel', COLORS[1]), ('f1', 'F1', COLORS[2])]:
        figure.add_trace(go.Bar(x=[r['label'] for r in rows], y=[r[metric] for r in rows],
                                name=name, marker_color=color))
    figure.update_yaxes(range=[0, 1], tickformat='.0%')
    figure.update_layout(barmode='group')
    return style(figure, 'Résultats par classe')


def comparison_chart(histories, metric):
    figure = go.Figure()
    for index, (name, history) in enumerate(histories.items()):
        rows = epoch_records(history)
        figure.add_trace(go.Scatter(x=[r['epoch'] for r in rows],
                                    y=[(r.get('validation') or r['train'])[metric] for r in rows],
                                    mode='lines', name=name,
                                    line=dict(color=COLORS[index % len(COLORS)], width=2)))
    figure.update_xaxes(title='Époque')
    if metric == 'accuracy':
        figure.update_yaxes(tickformat='.0%', range=[0, 1])
    return style(figure, 'Comparaison • ' + ('loss' if metric == 'loss' else 'accuracy'))


def history_csv(history):
    output = StringIO()
    fields = ['epoch', 'train_loss', 'validation_loss', 'train_accuracy',
              'validation_accuracy', 'best_epoch', 'best_loss', 'elapsed_seconds']
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in epoch_records(history):
        writer.writerow({'epoch': row['epoch'], 'train_loss': row['train']['loss'],
                         'validation_loss': row['validation']['loss'] if row.get('validation') else '',
                         'train_accuracy': row['train']['accuracy'],
                         'validation_accuracy': row['validation']['accuracy'] if row.get('validation') else '',
                         'best_epoch': row['best_epoch'], 'best_loss': row['best_loss'],
                         'elapsed_seconds': row['elapsed_seconds']})
    return output.getvalue()


def figures_html(figures):
    parts = ['<!doctype html><html lang="fr"><meta charset="utf-8">',
             '<title>My Torch — rapport</title><body style="background:#0a1120;color:white">',
             '<h1>My Torch — rapport d’expérience</h1>']
    for index, figure in enumerate(figures):
        parts.append(figure.to_html(full_html=False, include_plotlyjs=True if index == 0 else False))
    parts.append('</body></html>')
    return '\n'.join(parts)
