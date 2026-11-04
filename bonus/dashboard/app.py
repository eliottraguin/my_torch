"""Launch with: make dashboard (from the project root)."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / 'src'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import chess
import chess.svg
import pandas as pd
import streamlit as st

from analyzer.chess.fen import FEN
from bonus.dashboard.charts import (
    COLORS, comparison_chart, confusion_chart, distribution_chart, epoch_records,
    figures_html, history_csv, learning_chart, per_class_chart, style,
)
from bonus.dashboard.engine import (
    RUNS, TrainingJob, TrainingOptions, evaluate_positions, load_network,
    model_info, predictions_csv, profile_dataset, read_history, read_positions,
)

st.set_page_config(page_title='My Torch • Chess Lab', page_icon='♞', layout='wide')
st.markdown('''<style>
[data-testid="stAppViewContainer"] {background:#0a1120;color:#e5edf8;}
[data-testid="stHeader"] {background:rgba(10,17,32,.9);}
[data-testid="stSidebar"] {background:#101a2c;border-right:1px solid #263349;}
[data-testid="stMetric"] {background:#111b2c;border:1px solid #263349;border-radius:14px;padding:16px;}
[data-testid="stMetricLabel"] {color:#aabbd1;}
[data-testid="stMetricValue"] {color:#f4f7fb;}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {font-size:1.25rem;}
.stButton>button[kind="primary"] {background:#52dacb;color:#06201e;border:0;font-weight:700;}
h1,h2,h3 {letter-spacing:-.025em;}
.hero {padding:22px 0 30px;} .eyebrow {color:#52dacb;letter-spacing:.18em;font-size:12px;font-weight:700;}
.hero h1 {font-size:42px;margin:6px 0;} .hero p {color:#aabbd1;font-size:16px;}
[data-testid="stTabs"] [role="tablist"] {gap:25px;border-bottom:1px solid #263349;}
[data-testid="stTabs"] [aria-selected="true"] {color:#52dacb;}
</style>''', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_info(path, modified):
    return model_info(path)


@st.cache_data(show_spinner=False)
def cached_profile(path, modified, limit):
    return profile_dataset(path, limit)


@st.cache_data(show_spinner=False, max_entries=8)
def training_report(history, logarithmic):
    return figures_html([learning_chart(history, logarithmic=logarithmic),
                         learning_chart(history, 'accuracy')])


def models():
    paths = list(ROOT.glob('*.nn'))
    if RUNS.exists():
        paths += list(RUNS.glob('*/model.nn'))
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def datasets():
    paths = list((ROOT / 'dataset').rglob('*.txt')) if (ROOT / 'dataset').exists() else []
    paths += list((ROOT / 'tests/dataset_example').rglob('*.txt'))
    return sorted(paths)


def display_path(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def choose_file(label, paths, key, preferred=''):
    strings = [str(path) for path in paths]
    index = next((i for i, value in enumerate(strings) if value.endswith(preferred)), 0) if preferred else 0
    return st.selectbox(label, strings, index=index if strings else None,
                        format_func=display_path, key=key)


def pct(value):
    return '—' if value is None else f'{100 * value:.1f} %'


def show_model(path):
    if not path:
        return
    try:
        info = cached_info(path, Path(path).stat().st_mtime_ns)
        st.caption(' → '.join(map(str, info['layers'])))
        a, b = st.columns(2)
        a.metric('Paramètres', f"{info['parameters']:,}".replace(',', ' '))
        b.metric('Classes', len(info['labels']))
        st.caption(f"{info['hidden_activation']} · sortie {info['output_activation']}")
    except Exception as error:
        st.error(f'Modèle illisible : {error}')


with st.sidebar:
    st.markdown('## ♞ MY TORCH')
    st.caption('CHESS LAB / EXPÉRIENCES LOCALES')
    st.divider()
    inspected_model = choose_file('Modèle à inspecter', models(), 'sidebar_model')
    show_model(inspected_model)
    st.divider()
    st.caption('Les expériences sont conservées dans .dashboard_runs/. Les modèles sources sont préservés.')
    st.caption('Les scores softmax ne sont pas des probabilités calibrées.')

st.markdown('''<div class="hero"><div class="eyebrow">MY TORCH / CHESS LAB</div>
<h1>Voir ce que le réseau apprend.</h1>
<p>Entraîner, mesurer et comprendre les prédictions sur tes échiquiers.</p></div>''', unsafe_allow_html=True)
train_tab, predict_tab, compare_tab, data_tab = st.tabs(['Entraînement', 'Prédiction', 'Comparer les modèles', 'Explorer les données'])


def show_history(history, prefix):
    records = epoch_records(history)
    if not records:
        st.info('Les courbes apparaîtront après la première évaluation. L’époque 0 mesure le modèle initial.')
        return
    last = records[-1]
    if not last.get('validation'):
        st.warning('Aucune validation distincte : ces métriques décrivent uniquement les données d’entraînement.')
    metric = last.get('validation') or last['train']
    a, b, c, d = st.columns(4)
    a.metric('Époque évaluée', f"{last['epoch']} / {last['max_epochs']}")
    b.metric('Accuracy surveillée', pct(metric['accuracy']))
    c.metric('Meilleure loss', f"{last['best_loss']:.5f}")
    d.metric('Durée', f"{last['elapsed_seconds']:.1f} s")
    log = st.checkbox('Échelle logarithmique pour la loss', key=prefix + '_log')
    loss_fig = learning_chart(history, logarithmic=log)
    accuracy_fig = learning_chart(history, 'accuracy')
    left, right = st.columns(2)
    left.plotly_chart(loss_fig, width='stretch', key=prefix + '_loss')
    right.plotly_chart(accuracy_fig, width='stretch', key=prefix + '_accuracy')
    st.caption('Courbes mesurées à poids fixes. Les métriques détaillées portent sur la dernière époque évaluée ; le modèle final peut être restauré à une époque antérieure.')
    a, b = st.columns(2)
    a.plotly_chart(confusion_chart(metric['confusion'], [r['label'] for r in metric['per_class']]),
                   width='stretch', key=prefix + '_confusion')
    b.plotly_chart(per_class_chart(metric['per_class']), width='stretch', key=prefix + '_classes')
    st.dataframe(pd.DataFrame(metric['per_class']), hide_index=True, width='stretch')
    a, b = st.columns(2)
    a.download_button('Télécharger les métriques CSV', history_csv(history),
                      file_name='training_metrics.csv', mime='text/csv', key=prefix + '_csv')
    b.download_button('Exporter les courbes HTML', training_report(history, log),
                      file_name='training_report.html', mime='text/html', key=prefix + '_html')


with train_tab:
    st.subheader('Nouvelle expérience')
    st.caption('Un entraînement s’exécute en arrière-plan. Tu peux explorer les autres onglets pendant son exécution.')
    mode = st.radio('Point de départ', ['Réseau vierge', 'Modèle existant'], horizontal=True)
    model_path = choose_file('Réseau de départ', models(), 'training_model') if mode == 'Modèle existant' else ''
    with st.form('training_form'):
        a, b = st.columns(2)
        training_choices = [str(p) for p in datasets()]
        default_train = next((i for i, p in enumerate(training_choices) if p.endswith('generated/train.txt')), 0)
        train_path = a.selectbox('Fichier d’entraînement', training_choices,
                                 index=default_train if training_choices else None,
                                 format_func=display_path, key='training_data')
        validation_choices = [''] + [str(p) for p in datasets()]
        default_validation = next((i for i, p in enumerate(validation_choices) if p.endswith('generated/validation.txt')), 0)
        validation_path = b.selectbox('Validation séparée (facultative)', validation_choices,
                                      index=default_validation,
                                      format_func=lambda p: display_path(p) if p else 'Découpage interne à 20 %')
        a, b, c, d = st.columns(4)
        epochs = a.number_input('Époques maximum', 1, 2000, 100)
        learning_rate = b.number_input('Learning rate', min_value=0.000001, max_value=10.0, value=0.01, format='%.6f')
        patience = c.number_input('Patience', 1, 200, 10)
        seed = d.number_input('Graine', min_value=0, value=42)
        a, b = st.columns(2)
        min_delta = a.number_input('Amélioration minimale', min_value=0.0, value=0.000001, format='%.8f')
        limit = b.number_input('Limite de lignes par fichier (0 = toutes)', min_value=0, value=0, step=100)
        with st.expander('Architecture du réseau vierge', expanded=mode == 'Réseau vierge'):
            a, b, c = st.columns(3)
            hidden = a.text_input('Couches cachées', '128,64')
            activation = b.selectbox('Activation cachée', ['relu', 'silu', 'sigmoid'])
            output = c.selectbox('Activation de sortie', ['softmax', 'sigmoid', 'relu', 'silu'])
            classes = st.selectbox('Classes', ['5 classes avec couleurs', '3 classes'])
            init = st.number_input('Amplitude de l’initialisation uniforme', min_value=0.001, value=0.1)
        active = st.session_state.get('job')
        running = active is not None and active.snapshot()['status'] == 'running'
        launch = st.form_submit_button('Lancer l’entraînement', type='primary', disabled=running or not train_path)
    if launch:
        try:
            if mode == 'Modèle existant' and not model_path:
                raise ValueError('Choisis un modèle de départ.')
            labels = ['Checkmate White', 'Checkmate Black', 'Check White', 'Check Black', 'Nothing'] if classes.startswith('5') else ['Checkmate', 'Check', 'Nothing']
            sizes = [int(value.strip()) for value in hidden.split(',') if value.strip()]
            if any(size < 1 or size > 4096 for size in sizes):
                raise ValueError('Les couches cachées doivent contenir entre 1 et 4096 neurones.')
            config = {'name': 'dashboard', 'layers': [838] + sizes + [len(labels)],
                      'labels': labels, 'seed': int(seed), 'init': float(init),
                      'learning_rate': float(learning_rate), 'hidden_activation': activation,
                      'output_activation': output}
            options = TrainingOptions(train_path=train_path, validation_path=validation_path,
                                      model_path=model_path or '', epochs=int(epochs),
                                      learning_rate=float(learning_rate), patience=int(patience),
                                      min_delta=float(min_delta), seed=int(seed), limit=int(limit))
            st.session_state.job = TrainingJob(options, config).start()
            st.session_state.last_job_status = 'running'
            st.rerun()
        except Exception as error:
            st.error(str(error))

    @st.fragment(run_every=1.0)
    def live_training():
        job = st.session_state.get('job')
        if job is None:
            st.info('Lance une expérience pour voir ses courbes en direct.')
            return
        snapshot = job.snapshot()
        status = snapshot['status']
        previous_status = st.session_state.get('last_job_status')
        st.session_state.last_job_status = status
        if previous_status == 'running' and status != 'running':
            st.rerun()
        captions = {'running': 'En cours', 'completed': 'Terminé • meilleur modèle restauré',
                    'stopped': 'Arrêté • dernière époque complète conservée', 'error': 'Échec'}
        st.subheader(captions.get(status, status))
        st.caption(display_path(snapshot['directory']))
        records = epoch_records(snapshot['history'])
        if status == 'running':
            if st.button('Arrêter et sauvegarder', key='stop_training'):
                job.stop()
                st.info('Arrêt demandé. Le checkpoint de la dernière époque complète sera sauvegardé.')
            fraction = records[-1]['epoch'] / records[-1]['max_epochs'] if records else 0.0
            st.progress(min(1.0, fraction), text='Progression par rapport au maximum d’époques')
        if snapshot['error']:
            st.error(snapshot['error'])
        config_path = Path(snapshot['directory']) / 'config.json'
        if config_path.exists():
            configuration = json.loads(config_path.read_text())
            skipped = len(configuration['skipped_train']) + len(configuration['skipped_validation'])
            if skipped:
                st.warning(f'{skipped} lignes invalides ont été exclues. Détails dans la configuration exportée.')
            if len(configuration.get('train_classes', {})) < 2:
                st.warning('Une seule classe est présente dans les données d’entraînement.')
            st.download_button('Configuration de l’expérience', config_path.read_bytes(),
                               file_name='experiment.json', key='download_configuration')
        show_history(snapshot['history'], 'live')
        saved = Path(snapshot['directory']) / 'model.nn'
        if saved.exists() and status != 'running':
            st.download_button('Télécharger le réseau .nn', saved.read_bytes(),
                               file_name='my_torch_network_dashboard.nn', key='download_model')
    live_training()

with predict_tab:
    st.subheader('Évaluer les prédictions')
    selected = choose_file('Modèle à évaluer', models(), 'prediction_model')
    input_mode = st.radio('Source des positions', ['Fichier FEN', 'Saisie FEN'], horizontal=True)
    input_path = choose_file('Fichier à analyser', datasets(), 'prediction_data', 'generated/test.txt') if input_mode == 'Fichier FEN' else None
    fen_text = st.text_area('Une position FEN par ligne, label facultatif',
                            '4k3/8/8/8/8/8/4R3/4K3 b - - 0 12 Check White') if input_mode == 'Saisie FEN' else ''
    limit_prediction = st.number_input('Positions maximum (0 = toutes)', min_value=0, value=1000, step=100)
    if st.button('Analyser les positions', type='primary', disabled=not selected):
        try:
            with st.spinner('Prédictions en cours…'):
                if input_mode == 'Fichier FEN':
                    if not input_path:
                        raise ValueError('Choisis un fichier FEN.')
                    positions, errors = read_positions(input_path, int(limit_prediction))
                else:
                    lines = [line.strip() for line in fen_text.splitlines() if line.strip()]
                    if limit_prediction:
                        lines = lines[:int(limit_prediction)]
                    positions = [FEN(line).parse() for line in lines]
                    if not positions:
                        raise ValueError('Saisis au moins une position.')
                    errors = []
                result = evaluate_positions(load_network(selected), positions)
                result.update(model_path=selected, errors=errors)
                st.session_state.prediction_result = result
        except Exception as error:
            st.error(str(error))
    result = st.session_state.get('prediction_result')
    if result:
        st.caption(f"Résultats de {display_path(result['model_path'])} — {result['total']} positions analysées")
        a, b, c, d = st.columns(4)
        a.metric('Accuracy', pct(result['accuracy']))
        b.metric('F1 macro', pct(result['macro_f1']))
        c.metric('Bonnes réponses', f"{result['correct']} / {result['count']}")
        d.metric('Sans score', result['unscored'])
        if not result['count']:
            st.info('Ajoute des labels attendus compatibles pour calculer un taux de réussite.')
        if result['unknown_labels'] or result['errors']:
            st.warning(f"Labels inconnus exclus : {result['unknown_labels']}. Lignes invalides exclues : {len(result['errors'])}.")
        labels = [row['label'] for row in result['per_class']]
        normalize = st.checkbox('Normaliser la matrice par classe attendue')
        a, b = st.columns(2)
        a.plotly_chart(confusion_chart(result['confusion'], labels, normalize), width='stretch')
        b.plotly_chart(per_class_chart(result['per_class']), width='stretch')
        st.dataframe(pd.DataFrame(result['per_class']), hide_index=True, width='stretch')
        only_errors = st.checkbox('Afficher uniquement les erreurs de classification')
        rows = [r for r in result['rows'] if r['correct'] is False] if only_errors else result['rows']
        st.dataframe(pd.DataFrame([{k: v for k, v in row.items() if k != 'scores'} for row in rows]),
                     hide_index=True, width='stretch')
        st.download_button('Exporter les prédictions CSV', predictions_csv(result['rows']),
                            file_name='predictions.csv', mime='text/csv')
        st.download_button('Exporter le rapport HTML',
                            figures_html([confusion_chart(result['confusion'], labels), per_class_chart(result['per_class'])]),
                            file_name='predictions_report.html', mime='text/html')
        if rows:
            index = st.selectbox('Inspecter une position', range(len(rows)),
                                  format_func=lambda i: f"#{rows[i]['position']} · {rows[i]['prediction']}")
            row = rows[index]
            a, b = st.columns([1, 2])
            board = chess.Board(' '.join(row['fen'].split()[:6]))
            a.image(chess.svg.board(board, size=300))
            b.code(row['fen'], language=None)
            b.caption('Scores de sortie • ' + result['output_activation'] + ' • non calibrés')
            b.plotly_chart(distribution_chart(row['scores'], 'Scores par classe'), width='stretch')

with compare_tab:
    st.subheader('Historique des expériences')
    run_paths = sorted([p for p in RUNS.glob('*') if p.is_dir() and (p / 'history.jsonl').exists()], reverse=True) if RUNS.exists() else []
    chosen = st.multiselect('Expériences à comparer', [str(p) for p in run_paths],
                            format_func=lambda p: Path(p).name)
    histories = {Path(path).name: read_history(path) for path in chosen}
    if histories:
        a, b = st.columns(2)
        first, second = comparison_chart(histories, 'loss'), comparison_chart(histories, 'accuracy')
        a.plotly_chart(first, width='stretch')
        b.plotly_chart(second, width='stretch')
        comparison = []
        for path in chosen:
            history = read_history(path)
            records = epoch_records(history)
            if records:
                best = min(records, key=lambda r: (r.get('validation') or r['train'])['loss'])
                metrics = best.get('validation') or best['train']
                comparison.append({'expérience': Path(path).name, 'meilleure époque observée': best['epoch'],
                                   'loss': metrics['loss'], 'accuracy': metrics['accuracy'],
                                   'F1 macro': metrics['macro_f1'],
                                   'ensemble': 'validation' if best.get('validation') else 'entraînement'})
        st.dataframe(pd.DataFrame(comparison), hide_index=True, width='stretch')
        st.caption('Compare des expériences utilisant les mêmes données et la même séparation. La validation sert au choix du modèle ; réserve le test pour le bilan final.')
        st.download_button('Exporter la comparaison HTML', figures_html([first, second]),
                            file_name='comparison.html', mime='text/html')
    else:
        st.info('Les expériences lancées depuis ce dashboard sont conservées ici. Sélectionne-en plusieurs pour superposer leurs courbes.')

with data_tab:
    st.subheader('Comprendre les données')
    data_path = choose_file('Dataset à explorer', datasets(), 'profile_data', 'generated/train.txt')
    sample_limit = st.number_input('Lignes à inspecter (0 = toutes)', min_value=0, value=10000, step=1000)
    if st.button('Calculer le profil', disabled=not data_path):
        try:
            with st.spinner('Lecture des données…'):
                profile = cached_profile(data_path, Path(data_path).stat().st_mtime_ns, int(sample_limit))
                st.session_state.profile = dict(profile, path=data_path)
        except Exception as error:
            st.error(str(error))
    profile = st.session_state.get('profile')
    if profile:
        st.caption(f"{display_path(profile['path'])} · profil limité à {profile['limit']} lignes" if profile['limit'] else display_path(profile['path']))
        a, b = st.columns(2)
        a.metric('Positions exploitables inspectées', profile['count'])
        b.metric('Lignes invalides', len(profile['errors']))
        a, b = st.columns(2)
        a.plotly_chart(distribution_chart(profile['labels'], 'Répartition des classes'), width='stretch')
        b.plotly_chart(distribution_chart(dict(sorted(profile['material'].items())), 'Nombre de pièces'), width='stretch')
        st.plotly_chart(distribution_chart(profile['turns'], 'Joueur au trait'), width='stretch')
        if profile['errors']:
            st.dataframe(pd.DataFrame(profile['errors']), hide_index=True)
