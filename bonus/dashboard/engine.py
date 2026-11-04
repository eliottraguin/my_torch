"""Local experiment storage, training jobs and prediction evaluation."""

from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import threading
import uuid

import numpy as np

from analyzer.chess.chessencoder import ChessEncoder
from analyzer.chess.fen import FEN
from analyzer.metrics import classification_metrics
from analyzer.neural_network import NeuralNetwork
from analyzer.training import run_training
from generator.neuralfile import NeuralFile

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / '.dashboard_runs'


def save_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_history(directory):
    path = Path(directory) / 'history.jsonl'
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A writer may be appending its last line.
    return records


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_model(network):
    sizes = [network.input_size] + network.hidden_layers + [network.output_size]
    if (network.input_size != 838 or len(network.labels) != network.output_size
            or len(set(network.labels)) != network.output_size):
        raise ValueError('Le modèle doit avoir 838 entrées et un label unique par sortie.')
    if len(network.weights) != len(sizes) - 1 or len(network.biases) != len(network.weights):
        raise ValueError('Nombre de matrices ou de biais incohérent.')
    for index, (weights, biases) in enumerate(zip(network.weights, network.biases)):
        if weights.shape != (sizes[index + 1], sizes[index]) or biases.shape != (sizes[index + 1],):
            raise ValueError(f'Dimensions incorrectes pour la couche {index + 1}.')
        if not np.isfinite(weights).all() or not np.isfinite(biases).all():
            raise ValueError('Le modèle contient des paramètres non finis.')
    return network


def load_network(path):
    return check_model(NeuralNetwork.load_from_file(path))


def model_info(path):
    network = load_network(path)
    return {'layers': [network.input_size] + network.hidden_layers + [network.output_size],
            'parameters': sum(w.size for w in network.weights) + sum(b.size for b in network.biases),
            'labels': network.labels, 'learning_rate': network.learning_rate,
            'hidden_activation': network.activation_function.func_name,
            'output_activation': network.output_activation_function.func_name}


def read_positions(path, limit=0):
    positions, errors = [], []
    with Path(path).open() as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if limit and len(positions) + len(errors) >= limit:
                break
            try:
                fen = FEN(line.strip()).parse()
                encoded = ChessEncoder(fen).encode()
                if not np.isfinite(encoded).all():
                    raise ValueError('Encodage non fini')
                positions.append(fen)
            except (ValueError, OverflowError) as error:
                errors.append({'line': index, 'error': str(error)})
    if not positions:
        raise ValueError('Aucune position exploitable dans ce fichier.')
    return positions, errors


def profile_dataset(path, limit=10000):
    positions, errors = read_positions(path, limit)
    labels = Counter(' '.join(p.label) if p.label else 'Sans label' for p in positions)
    material = Counter(sum(c.isalpha() for c in p.piece_placement) for p in positions)
    turns = Counter('Blancs' if p.active_color == 'w' else 'Noirs' for p in positions)
    return {'count': len(positions), 'errors': errors, 'labels': dict(labels),
            'material': dict(material), 'turns': dict(turns), 'limit': limit}


def evaluate_positions(network, positions):
    rows = []
    confusion = np.zeros((len(network.labels), len(network.labels)), dtype=np.int64)
    ignored_labels = Counter()
    for index, fen in enumerate(positions, 1):
        output = network.forward_pass(ChessEncoder(fen).encode())[-1]
        if not np.isfinite(output).all():
            raise ValueError('Sortie du réseau non finie.')
        predicted_index = int(np.argmax(output))
        prediction = network.labels[predicted_index]
        raw_label = ' '.join(fen.label) if fen.label else ''
        expected = raw_label
        # A three-output model can be evaluated against colored annotations.
        if expected not in network.labels and fen.label and fen.label[0] in network.labels:
            expected = fen.label[0]
        known = expected in network.labels
        if known:
            confusion[network.labels.index(expected), predicted_index] += 1
        elif expected:
            ignored_labels[expected] += 1
        rows.append({'position': index, 'fen': fen.fen_string,
                     'expected': raw_label, 'prediction': prediction,
                     'score': float(output[predicted_index]),
                     'correct': prediction == expected if known else None,
                     'scores': {label: float(score) for label, score in zip(network.labels, output)}})
    result = classification_metrics(confusion, network.labels)
    result.update(total=len(rows), unscored=len(rows) - result['count'],
                  unknown_labels=dict(ignored_labels), rows=rows,
                  output_activation=network.output_activation_function.func_name)
    return result


def predictions_csv(rows):
    stream = StringIO()
    columns = ['position', 'fen', 'expected', 'prediction', 'score', 'correct']
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


@dataclass(frozen=True)
class TrainingOptions:
    train_path: str
    validation_path: str = ''
    model_path: str = ''
    epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-6
    learning_rate: float = 0.01
    seed: int = 42
    limit: int = 0


class TrainingJob:
    """Run NumPy in a worker; all UI interaction stays in Streamlit's main thread."""
    def __init__(self, options, config=None, output_root=RUNS):
        self.options = options
        self.config = config
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        self.directory = Path(output_root) / f'{stamp}-{uuid.uuid4().hex[:6]}'
        self.stop_requested = threading.Event()
        self.lock = threading.Lock()
        self.status = 'created'
        self.error = None
        self.thread = None

    def start(self):
        with self.lock:
            if self.thread is not None:
                raise ValueError('Cette expérience a déjà démarré.')
            self.directory.mkdir(parents=True, exist_ok=False)
            self.status = 'running'
            self.thread = threading.Thread(target=self._work, daemon=True)
            self.thread.start()
        return self

    def stop(self):
        self.stop_requested.set()

    def snapshot(self):
        with self.lock:
            return {'status': self.status, 'error': self.error,
                    'directory': str(self.directory), 'history': read_history(self.directory)}

    def _observe(self, record):
        with (self.directory / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        if record['event'] in ('completed', 'stopped'):
            self._result = record['event']

    def _work(self):
        self._result = 'completed'
        try:
            options = self.options
            if not np.isfinite(options.learning_rate) or options.learning_rate <= 0:
                raise ValueError('Le taux d’apprentissage doit être positif et fini.')
            if options.model_path:
                network = load_network(options.model_path)
            else:
                if self.config is None:
                    raise ValueError('Configuration du réseau manquante.')
                from types import SimpleNamespace
                data = NeuralFile(SimpleNamespace(get_config=lambda: self.config)).get_data()
                save_json(self.directory / 'initial.nn', data)
                network = load_network(self.directory / 'initial.nn')
            network.learning_rate = options.learning_rate
            # Every experiment gets an independent model, including resumed ones.
            network.save(self.directory / 'initial.nn')
            positions, errors = read_positions(options.train_path, options.limit)
            validation, validation_errors = (read_positions(options.validation_path, options.limit)
                                             if options.validation_path else (None, []))
            classes = Counter(' '.join(p.label) for p in positions if p.label)
            unknown = set(classes) - set(network.labels)
            if unknown:
                raise ValueError(f'Labels incompatibles avec le modèle : {sorted(unknown)}')
            if validation:
                unknown_validation = {' '.join(p.label) for p in validation if p.label} - set(network.labels)
                if unknown_validation:
                    raise ValueError(f'Labels de validation incompatibles : {sorted(unknown_validation)}')
            metadata = dict(vars(options), model=model_info(self.directory / 'initial.nn'),
                            train_classes=dict(classes),
                            skipped_train=errors, skipped_validation=validation_errors,
                            train_sha256=fingerprint(options.train_path),
                            validation_sha256=fingerprint(options.validation_path) if options.validation_path else None,
                            initial_model_sha256=fingerprint(self.directory / 'initial.nn'))
            save_json(self.directory / 'config.json', metadata)
            run_training(network, positions, self.directory / 'model.nn',
                         validation_positions=validation, patience=options.patience,
                         min_delta=options.min_delta, seed=options.seed,
                         max_epochs=options.epochs, observer=self._observe,
                         should_stop=self.stop_requested.is_set)
            with self.lock:
                self.status = self._result
        except Exception as error:
            with self.lock:
                self.status = 'error'
                self.error = str(error)
        finally:
            with self.lock:
                save_json(self.directory / 'result.json', {'status': self.status, 'error': self.error})
