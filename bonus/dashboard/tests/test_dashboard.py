"""Dashboard workers, prediction scoring, reports and actual Streamlit controls."""

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from analyzer.chess.fen import FEN
from bonus.dashboard.charts import confusion_chart, figures_html, learning_chart
from bonus.dashboard.engine import (
    ROOT, TrainingJob, TrainingOptions, evaluate_positions, load_network,
    model_info, predictions_csv, profile_dataset, read_history,
)
from tests.support import write_test_network


class DashboardTests(unittest.TestCase):
    def test_inspection_accepts_legacy_objects_without_training_attributes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'old-model.nn'
            write_test_network(path)
            network = load_network(path)
            for name in ('max_epochs', 'patience', 'min_delta'):
                delattr(network, name)
            with patch('bonus.dashboard.engine.load_network', return_value=network):
                info = model_info(path)
            self.assertEqual((info['max_epochs'], info['patience'], info['min_delta']),
                             (200, 10, 1e-6))

    def test_legacy_loader_preserves_custom_settings_from_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'model.nn'
            write_test_network(path)
            network = load_network(path)
            for name in ('max_epochs', 'patience', 'min_delta'):
                delattr(network, name)
            data = json.loads(path.read_text())
            data.update(max_epochs=37, patience=4, min_delta=.0002)
            path.write_text(json.dumps(data))
            with patch('bonus.dashboard.engine.NeuralNetwork.load_from_file', return_value=network):
                info = model_info(path)
            self.assertEqual((info['max_epochs'], info['patience'], info['min_delta']),
                             (37, 4, .0002))

    def test_prediction_metrics_exclude_unknown_and_missing_labels(self):
        with TemporaryDirectory() as directory:
            model = Path(directory) / 'model.nn'
            write_test_network(model)
            positions = [FEN('8/8/8/8/8/8/8/K6k w - - 0 1 ' + label).parse()
                         for label in ('Check White', 'Nothing', 'Unknown', '')]
            result = evaluate_positions(load_network(model), positions)
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['unscored'], 2)
        self.assertIn('Unknown', result['unknown_labels'])
        self.assertEqual(len(predictions_csv(result['rows']).splitlines()), 5)

    def test_real_worker_saves_history_configuration_and_independent_model(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            directory = Path(directory)
            source, boards = directory / 'source.nn', directory / 'boards.txt'
            write_test_network(source)
            original = source.read_bytes()
            boards.write_text('8/8/8/8/8/8/8/K6k w - - 0 1 Nothing\n')
            options = TrainingOptions(train_path=str(boards), model_path=str(source), epochs=2)
            job = TrainingJob(options, output_root=directory / 'runs').start()
            job.thread.join(timeout=10)
            self.assertFalse(job.thread.is_alive())
            snapshot = job.snapshot()
            self.assertEqual(snapshot['status'], 'completed', snapshot['error'])
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue((job.directory / 'model.nn').exists())
            self.assertEqual(len([r for r in read_history(job.directory) if r['event'] == 'epoch']), 3)
            config = json.loads((job.directory / 'config.json').read_text())
            self.assertIn('train_sha256', config)

    def test_fresh_worker_and_invalid_input_report_status(self):
        config = {'name': 'test', 'layers': [838, 4, 3], 'labels': ['Checkmate', 'Check', 'Nothing'],
                  'learning_rate': .01, 'seed': 42, 'init': .1,
                  'hidden_activation': 'relu', 'output_activation': 'softmax'}
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            boards = Path(directory) / 'boards.txt'
            boards.write_text('8/8/8/8/8/8/8/K6k w - - 0 1 Nothing\n')
            job = TrainingJob(TrainingOptions(str(boards), epochs=1), config, Path(directory) / 'runs').start()
            job.thread.join(10)
            self.assertEqual(job.snapshot()['status'], 'completed', job.snapshot()['error'])
            bad = TrainingJob(TrainingOptions(str(boards) + '.missing'), config, Path(directory) / 'runs').start()
            bad.thread.join(10)
            self.assertEqual(bad.snapshot()['status'], 'error')
            self.assertTrue((bad.directory / 'result.json').exists())

    def test_dataset_profile_excludes_invalid_records(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'data.txt'
            path.write_text('8/8/8/8/8/8/8/K6k w - - 0 1 Nothing\ninvalid\n')
            profile = profile_dataset(path)
        self.assertEqual(profile['count'], 1)
        self.assertEqual(len(profile['errors']), 1)
        self.assertEqual(profile['labels'], {'Nothing': 1})

    def test_html_reports_include_plotly_and_confusion_axes(self):
        figure = confusion_chart([[1, 0], [1, 2]], ['a', 'b'], normalize=True)
        document = figures_html([figure, learning_chart([])])
        self.assertIn('plotly', document.lower())
        self.assertNotIn('<script src=', document)
        self.assertEqual(figure.layout.xaxis.title.text, 'Classe prédite')
        self.assertEqual(figure.layout.yaxis.title.text, 'Classe attendue')

    def test_streamlit_renders_four_tabs_and_runs_single_position_prediction(self):
        with TemporaryDirectory() as directory:
            model = Path(directory) / 'ui-model.nn'
            write_test_network(model)
            network = load_network(model)
            # UI behavior must not depend on a user's model being edited or trained.
            with patch('bonus.dashboard.engine.load_network', return_value=network):
                self.check_ui_prediction()

    def check_ui_prediction(self):
        app = AppTest.from_file(str(ROOT / 'bonus/dashboard/app.py'), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 4)
        mode = next(r for r in app.radio if r.label == 'Source des positions')
        mode.set_value('Saisie FEN').run()
        models = next(s for s in app.selectbox if s.label == 'Modèle à évaluer')
        if not models.options:
            self.skipTest('No delivered model available for the UI prediction smoke test')
        next(b for b in app.button if b.label == 'Analyser les positions').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(m.label == 'Accuracy' for m in app.metric))
