#!/usr/bin/env python3
"""Build reproducible, verified chess training datasets from FEN source files."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

try:
    import chess
except ImportError:
    print("Error: install dataset dependencies with: "
          "python -m pip install -r scripts/requirements-dataset.txt", file=sys.stderr)
    raise SystemExit(84) from None


LABELS = ("Checkmate White", "Checkmate Black", "Check White", "Check Black", "Nothing")
SPLITS = ("train", "validation", "test")
BINS = ("2-10", "11-20", "21-32")


@dataclass(frozen=True)
class Example:
    fen: str
    label: str
    material: str

    @property
    def line(self):
        return f"{self.fen} {self.label}\n"


def classify(board, classes=5):
    """Color denotes the attacking/winning side, opposite the player to move."""
    state = "Checkmate" if board.is_checkmate() else "Check" if board.is_check() else "Nothing"
    if classes == 3 or state == "Nothing":
        return state
    return f"{state} {'Black' if board.turn == chess.WHITE else 'White'}"


def material_bin(board):
    count = chess.popcount(board.occupied)
    return BINS[0] if count <= 10 else BINS[1] if count <= 20 else BINS[2]


def position_key(board):
    """Deduplicate a game state across different halfmove/fullmove counters."""
    return " ".join(board.fen(en_passant="fen").split()[:4])


def family_key(board):
    """Group identical placements and their vertical/color mirrors, ignoring metadata."""
    return _family_from_placement(board.board_fen(), board.turn)


def _family_from_placement(placement, turn):
    original = f"{placement} {'w' if turn else 'b'}"
    mirrored = '/'.join(reversed(placement.split('/'))).swapcase()
    reflected = f"{mirrored} {'b' if turn else 'w'}"
    return min(original, reflected)


def assign_split(board, seed, validation_fraction, test_fraction):
    # A family always belongs to one split, even if clocks/castling/EP differ.
    return _split_for_family(family_key(board), seed, validation_fraction, test_fraction)


def _split_for_family(family, seed, validation_fraction, test_fraction):
    digest = hashlib.blake2b(f"{seed}:{family}".encode(), digest_size=8).digest()
    fraction = int.from_bytes(digest, 'big') / (1 << 64)
    if fraction < test_fraction:
        return "test"
    if fraction < test_fraction + validation_fraction:
        return "validation"
    return "train"


def quotas(labels, per_class, validation_fraction, test_fraction, balance_material):
    bins = BINS if balance_material else ("all",)
    targets = {}
    for label in labels:
        for index, material in enumerate(bins):
            count = per_class // len(bins) + (index < per_class % len(bins))
            validation = max(1, round(count * validation_fraction))
            test = max(1, round(count * test_fraction))
            train = count - validation - test
            if train < 1:
                raise ValueError("Too few examples per stratum for the requested split fractions")
            for split, amount in zip(SPLITS, (train, validation, test)):
                targets[(split, label, material)] = amount
    return targets


def collect_examples(sources, targets, seed, classes, validation_fraction,
                     test_fraction, balance_material):
    """Audit every row and reservoir-sample each split/class/material stratum."""
    rng = random.Random(seed)
    pools = defaultdict(list)
    available = Counter()
    stats = Counter()
    rejected = []
    seen = set()
    provenance = []
    for source in sources:
        source_hash = hashlib.sha256()
        source_rows = 0
        with source.open('rb') as stream:
            for number, raw in enumerate(stream, 1):
                source_hash.update(raw)
                source_rows += 1
                stats['rows_read'] += 1
                if stats['rows_read'] % 100000 == 0:
                    print(f"Verified {stats['rows_read']} source rows...", file=sys.stderr)
                try:
                    fields = raw.decode('utf-8').split()
                    if not fields:
                        stats['blank_rows'] += 1
                        continue
                    if len(fields) < 6:
                        raise ValueError("Expected six FEN fields")
                    board = chess.Board(' '.join(fields[:6]))
                    if not board.is_valid():
                        raise ValueError(f"Invalid chess position (status={board.status()})")
                except (ValueError, UnicodeError) as error:
                    stats['rejected_rows'] += 1
                    if len(rejected) < 10:
                        rejected.append({'source': str(source), 'line': number, 'reason': str(error)})
                    continue
                label = classify(board, classes)
                if fields[6:]:
                    old_label = ' '.join(fields[6:]) if classes == 5 else fields[6]
                    stats['corrected_labels'] += old_label != label
                placement = board.board_fen()
                state = (f"{placement} {'w' if board.turn else 'b'} {board.castling_xfen()} "
                         f"{chess.square_name(board.ep_square) if board.ep_square is not None else '-'}")
                key = hashlib.blake2b(state.encode(), digest_size=32).digest()
                if key in seen:
                    stats['duplicate_rows_removed'] += 1
                    continue
                seen.add(key)
                stats['unique_valid_positions'] += 1
                split = _split_for_family(_family_from_placement(placement, board.turn),
                                         seed, validation_fraction, test_fraction)
                material = material_bin(board) if balance_material else 'all'
                bucket = (split, label, material)
                available[bucket] += 1
                pool = pools[bucket]
                capacity = targets[bucket]
                example = None
                if len(pool) < capacity:
                    example = Example(f'{state} {board.halfmove_clock} {board.fullmove_number}',
                                      label, material_bin(board))
                    pool.append(example)
                else:
                    index = rng.randrange(available[bucket])
                    if index < capacity:
                        example = Example(f'{state} {board.halfmove_clock} {board.fullmove_number}',
                                          label, material_bin(board))
                        pool[index] = example
        provenance.append({'path': str(source), 'sha256': source_hash.hexdigest(), 'rows': source_rows})
    missing = [(key, len(pools[key]), amount) for key, amount in targets.items()
               if len(pools[key]) < amount]
    if missing:
        details = '; '.join(f'{split}/{label}/{material}: {got}/{need}'
                            for (split, label, material), got, need in missing[:8])
        raise ValueError(f"Insufficient source diversity: {details}. Add sources, lower "
                         "--per-class, or use --no-material-balance. No output was written.")
    datasets = {split: [] for split in SPLITS}
    for bucket in sorted(targets):
        datasets[bucket[0]].extend(pools[bucket])
    for examples in datasets.values():
        rng.shuffle(examples)
    return datasets, {'source_files': provenance, 'audit': dict(stats),
                      'rejected_examples': rejected}


def describe(examples):
    result = {'rows': len(examples), 'labels': dict(Counter(e.label for e in examples)),
              'material': dict(Counter(e.material for e in examples))}
    turns = Counter()
    special = Counter()
    for example in examples:
        board = chess.Board(example.fen)
        turns['White' if board.turn else 'Black'] += 1
        special['castling_rights'] += bool(board.castling_rights)
        special['en_passant_field'] += board.ep_square is not None
        special['stalemate'] += board.is_stalemate()
    result.update(turns=dict(turns), special_positions=dict(special))
    return result


def write_datasets(destination, datasets, report):
    """Publish a complete new directory; never overwrite an existing dataset."""
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}; choose a new --output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.dataset-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'result'
        staging.mkdir()
        outputs = {}
        for split, examples in datasets.items():
            content = ''.join(example.line for example in examples).encode('utf-8')
            (staging / f'{split}.txt').write_bytes(content)
            outputs[f'{split}.txt'] = dict(describe(examples), sha256=hashlib.sha256(content).hexdigest())
        report['outputs'] = outputs
        (staging / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        staging.rename(destination)


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(84, f"{self.prog}: error: {message}\n")


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', type=Path, default=[Path('dataset/fusion.txt')],
                        help='FEN files, optionally followed by existing labels')
    parser.add_argument('--output', type=Path, default=Path('dataset/generated'))
    parser.add_argument('--per-class', type=int, default=12000,
                        help='Total per class across all three splits (default: 12000)')
    parser.add_argument('--classes', type=int, choices=(3, 5), default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--no-material-balance', action='store_true',
                        help='Do not require equal quantities of 2-10, 11-20 and 21-32 pieces')
    args = parser.parse_args(argv)
    fractions = (args.validation_fraction, args.test_fraction)
    if not all(math.isfinite(f) and 0 < f < 1 for f in fractions) or sum(fractions) >= 1:
        parser.error('validation/test fractions must be positive and their sum below one')
    if args.per_class < (10 if args.no_material_balance else 30):
        parser.error('--per-class must be at least 30 (10 with --no-material-balance)')
    try:
        if args.output.exists():
            raise FileExistsError(f'Output already exists: {args.output}; choose a new --output')
        sources = sorted({path.resolve(strict=True) for path in args.input})
        labels = LABELS if args.classes == 5 else ('Checkmate', 'Check', 'Nothing')
        targets = quotas(labels, args.per_class, *fractions, not args.no_material_balance)
        datasets, report = collect_examples(sources, targets, args.seed, args.classes,
                                            *fractions, not args.no_material_balance)
        report.update(schema_version=1, chess_version=chess.__version__, seed=args.seed,
                      classes=list(labels), per_class=args.per_class,
                      validation_fraction=fractions[0], test_fraction=fractions[1],
                      material_balance=not args.no_material_balance,
                      generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      split_policy='Stable hash of placement/turn and its color mirror; metadata ignored',
                      limitation='FEN files provide no game identifiers: distinct positions from one game may cross splits')
        write_datasets(args.output, datasets, report)
    except (OSError, ValueError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 84
    for split in SPLITS:
        print(f'{args.output / (split + ".txt")}: {len(datasets[split])} examples')
    print(f'Audit and reproduction settings: {args.output / "manifest.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
