"""Compare fresh networks using the project's SGD and validation-loss checkpoints."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from zoneinfo import ZoneInfo


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--configs', type=Path, help='Directory of generator .conf files')
    parser.add_argument('--screen-epochs', type=int, default=20)
    parser.add_argument('--final-epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--final-seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--baseline', help='Also confirm this candidate if not in the top two')
    args = parser.parse_args()
    if min(args.screen_epochs, args.final_epochs, args.patience) < 1:
        parser.error('Epoch limits and patience must be positive')
    if len(set(args.final_seeds)) != len(args.final_seeds) or min(args.final_seeds) < 0:
        parser.error('Final seeds must be distinct nonnegative integers')
    if args.output.exists():
        parser.error('Output directory already exists; choose a new directory')
    return args


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    args = arguments()
    root, out = args.root.resolve(), args.output.resolve()
    sys.path.insert(0, str(root / 'src'))
    import numpy as np
    from analyzer.chess.fen import FEN
    from analyzer.chess.chessencoder import ChessEncoder
    from analyzer.neural_network import NeuralNetwork
    from analyzer.training import train_network
    from analyzer.metrics import evaluate_metrics
    from analyzer.early_stopping import EarlyStopping
    from generator.configfile import ConfigFile
    from generator.neuralfile import NeuralFile

    base = json.loads((root / 'tests/conf_example/chess_network.conf').read_text())
    configs = {}
    def candidate(name, hidden, lr=.01, activation='relu', init=.1):
        configs[name] = {**base, 'name': 'my_torch_bench_' + name,
            'layers': [838, *hidden, 5], 'learning_rate': lr,
            'hidden_activation': activation, 'output_activation': 'softmax', 'init': init}
    if args.configs:
        configs = {p.stem: json.loads(p.read_text()) for p in sorted(args.configs.glob('*.conf'))}
    else:
        candidate('relu64', [64])
        candidate('relu128_64', [128, 64])
        candidate('relu256', [256])
        candidate('relu128_64_lr003', [128, 64], lr=.03)
        candidate('silu128_64_lr003', [128, 64], lr=.03, activation='silu')
        candidate('relu128_64_init02', [128, 64], init=.2)
        candidate('relu256_lr0003', [256], lr=.003)
        candidate('relu128_64_lr0003', [128, 64], lr=.003)
        candidate('relu128_64_lr0001', [128, 64], lr=.001)
    if len(configs) < 2:
        raise ValueError('At least two configurations are required')
    for config in configs.values():
        if config['layers'][0] != 838 or config['layers'][-1] != 5 or config['labels'] != base['labels']:
            raise ValueError('Configurations must share 838 inputs and five ordered labels')
        config.update(patience=args.patience, min_delta=1e-5)
    if args.baseline and args.baseline not in configs:
        raise ValueError('Unknown baseline')

    provenance, families = {}, {}
    def read_split(split):
        raw = (root / 'dataset/generated' / (split + '.txt')).read_bytes()
        groups, family = defaultdict(list), set()
        for line in raw.decode().splitlines():
            fields = line.split()
            placement, turn = fields[:2]
            reflected = '/'.join(reversed(placement.split('/'))).swapcase()
            family.add(min(placement + ' ' + turn, reflected + ' ' + ('b' if turn == 'w' else 'w')))
            label = ' '.join(fields[6:])
            if label not in base['labels']:
                raise ValueError(f'Unknown label: {label}')
            groups[label].append(line)
        families[split] = family
        provenance[split] = {'sha256': hashlib.sha256(raw).hexdigest(), 'count': sum(map(len, groups.values()))}
        return groups

    def sample(groups, count):
        rng, selected = random.Random(20261105), []
        for label in base['labels']:
            rows = groups[label].copy()
            rng.shuffle(rows)
            if len(rows) < count // 5:
                raise ValueError(f'Insufficient examples for {label}')
            selected.extend(rows[:count // 5])
        examples = []
        for line in selected:
            position = FEN(line).parse()
            target = np.zeros(5, dtype=np.float32)
            target[base['labels'].index(' '.join(position.label))] = 1
            encoded = ChessEncoder(position).encode()
            if encoded.shape != (838,) or not np.isfinite(encoded).all():
                raise ValueError('Invalid encoded position')
            examples.append((encoded, target))
        return examples

    train_groups, validation_groups = read_split('train'), read_split('validation')
    if families['train'] & families['validation']:
        raise ValueError('Train and validation overlap')
    train_small, train_large = sample(train_groups, 3000), sample(train_groups, 6000)
    validation = sample(validation_groups, 2000)
    results = {'date': datetime.now(ZoneInfo('Europe/Paris')).isoformat(),
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root / 'src').rglob('*.py'))},
        'protocol': {'numpy': np.__version__, 'python': sys.version,
            'blas_threads': os.environ.get('OPENBLAS_NUM_THREADS'),
            'screen': {'train': 3000, 'validation': 2000, 'epochs': args.screen_epochs, 'seeds': [42]},
            'finalists': {'train': 6000, 'validation': 2000, 'epochs': args.final_epochs, 'seeds': args.final_seeds},
            'selection': 'mean validation macro F1; checkpoint selected by validation MSE',
            'sampling_seed': 20261105, 'patience': args.patience, 'min_delta': 1e-5},
        'provenance': provenance, 'configs': configs, 'runs': []}
    out.mkdir(parents=True)

    def run(name, seed, train, epochs, stage):
        config = {**configs[name], 'seed': seed, 'max_epochs': epochs}
        config_path = out / f'{stage}_{name}_{seed}.conf'
        save_json(config_path, config)
        model_path = config_path.with_suffix('.nn')
        save_json(model_path, NeuralFile(ConfigFile(config_path)).get_data())
        network = NeuralNetwork.load_from_file(model_path)
        stopping, rng, examples = EarlyStopping(args.patience, 1e-5), random.Random(seed), list(train)
        started, history = time.monotonic(), []
        stopping.update(evaluate_metrics(network, validation)['loss'], network, 0)
        for epoch in range(1, epochs + 1):
            online_loss = train_network(network, examples, epoch, rng=rng)
            metrics = evaluate_metrics(network, validation)
            history.append({'epoch': epoch, 'online_loss': online_loss, 'validation': metrics})
            stop = stopping.update(metrics['loss'], network, epoch)
            if epoch % 10 == 0 or stop:
                print(f'{stage} {name} seed={seed} epoch={epoch} F1={metrics["macro_f1"]:.4f}', flush=True)
            if stop:
                break
        seconds = time.monotonic() - started
        stopping.restore(network)
        network.save(model_path)
        results['runs'].append({'stage': stage, 'name': name, 'seed': seed, 'epochs': epoch,
            'best_epoch': stopping.best_epoch, 'seconds': seconds,
            'parameters': sum(x.size for x in network.weights + network.biases),
            'train': evaluate_metrics(network, train), 'validation': evaluate_metrics(network, validation),
            'history': history})
        save_json(out / 'results.json', results)
        print('FINISHED', stage, name, seed, flush=True)

    for name in configs:
        run(name, 42, train_small, args.screen_epochs, 'screen')
    ranking = sorted(results['runs'], key=lambda r:r['validation']['macro_f1'], reverse=True)
    finalists = [r['name'] for r in ranking[:2]]
    if args.baseline and args.baseline not in finalists:
        finalists.append(args.baseline)
    results['finalists'] = finalists
    for name in finalists:
        for seed in args.final_seeds:
            run(name, seed, train_large, args.final_epochs, 'final')
    means = {name: float(np.mean([r['validation']['macro_f1'] for r in results['runs']
                                  if r['stage']=='final' and r['name']==name])) for name in finalists}
    winner = max(means, key=means.get)
    results.update(winner=winner, finalist_mean_f1=means)
    test_groups = read_split('test')
    if families['test'] & (families['train'] | families['validation']):
        raise ValueError('Test overlaps with train or validation')
    test = sample(test_groups, 6000)
    results['test'] = {str(seed): evaluate_metrics(NeuralNetwork.load_from_file(
        out / f'final_{winner}_{seed}.nn'), test) for seed in args.final_seeds}
    save_json(out / 'results.json', results)
    save_json(out / 'chess_recommended.conf', {**configs[winner], 'name':'my_torch_network_recommended',
                                             'seed':args.final_seeds[0], 'max_epochs':args.final_epochs})
    print('WINNER', winner, means, flush=True)


if __name__ == '__main__':
    main()
