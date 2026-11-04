"""Dataset label correctness, isolation, sampling and reproducibility."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import chess

from scripts.generate_dataset import (
    LABELS, SPLITS, Example, assign_split, classify, collect_examples,
    family_key, main, material_bin, position_key, quotas, write_datasets,
)


class DatasetGenerationTests(unittest.TestCase):
    def test_five_labels_use_attacking_color_and_mate_takes_precedence(self):
        mate = chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1')
        check = chess.Board('4k3/8/8/8/8/8/4R3/4K3 b - - 0 12')
        self.assertTrue(mate.is_valid())
        self.assertEqual(classify(mate), 'Checkmate White')
        self.assertEqual(classify(mate.mirror()), 'Checkmate Black')
        self.assertEqual(classify(check), 'Check White')
        self.assertEqual(classify(check.mirror()), 'Check Black')
        self.assertEqual(classify(chess.Board()), 'Nothing')
        self.assertEqual(classify(mate, 3), 'Checkmate')
        self.assertEqual(classify(check, 3), 'Check')

    def test_stalemate_belongs_to_nothing_in_current_three_and_five_class_models(self):
        board = chess.Board('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')
        self.assertTrue(board.is_stalemate())
        self.assertEqual(classify(board), 'Nothing')

    def test_mirrors_and_metadata_variants_stay_in_same_split(self):
        original = chess.Board()
        variant = original.copy()
        variant.halfmove_clock = 275
        variant.fullmove_number = 200
        variant.castling_rights = chess.BB_EMPTY
        self.assertEqual(family_key(original), family_key(original.mirror()))
        self.assertEqual(family_key(original), family_key(variant))
        for seed in range(20):
            expected = assign_split(original, seed, 0.1, 0.1)
            self.assertEqual(assign_split(original.mirror(), seed, 0.1, 0.1), expected)
            self.assertEqual(assign_split(variant, seed, 0.1, 0.1), expected)

    def test_dedup_key_ignores_counters_but_preserves_castling_rights(self):
        original = chess.Board()
        variant = original.copy()
        variant.halfmove_clock = 275
        variant.fullmove_number = 200
        self.assertEqual(position_key(original), position_key(variant))
        variant.castling_rights = chess.BB_EMPTY
        self.assertNotEqual(position_key(original), position_key(variant))

    def test_quota_totals_balance_all_classes_and_material(self):
        targets = quotas(LABELS, 12000, 0.1, 0.1, True)
        for label in LABELS:
            self.assertEqual(sum(n for (_, lbl, _), n in targets.items() if lbl == label), 12000)
        self.assertEqual(sum(n for (split, _, _), n in targets.items() if split == 'train'), 48000)
        self.assertEqual(material_bin(chess.Board()), '21-32')

    def source_for_all_splits(self, directory):
        board = chess.Board()
        rng = random.Random(7)
        examples = {}
        for _ in range(2000):
            if board.is_game_over():
                board.reset()
            board.push(rng.choice(list(board.legal_moves)))
            if classify(board) == 'Nothing':
                split = assign_split(board, 42, 0.1, 0.1)
                examples.setdefault(split, board.fen(en_passant='fen'))
            if len(examples) == 3:
                break
        self.assertEqual(len(examples), 3)
        path = Path(directory) / 'source.txt'
        rows = [f'{fen} Check White' for fen in examples.values()]
        repeated = chess.Board(next(iter(examples.values())))
        repeated.fullmove_number += 100
        rows += [f'{repeated.fen(en_passant="fen")} Nothing', '8/8/8/8/8/8/8/8 w - - 0 1 Nothing']
        path.write_text('\n'.join(rows) + '\n')
        return path

    def collect(self, path, capacity=1):
        targets = {(split, 'Nothing', 'all'): capacity for split in SPLITS}
        return collect_examples([path], targets, 42, 5, 0.1, 0.1, False)

    def test_collection_relabels_deduplicates_and_rejects_invalid_boards(self):
        with TemporaryDirectory() as directory:
            source = self.source_for_all_splits(directory)
            datasets, report = self.collect(source)
        self.assertEqual([len(datasets[s]) for s in SPLITS], [1, 1, 1])
        self.assertEqual(report['audit']['duplicate_rows_removed'], 1)
        self.assertEqual(report['audit']['rejected_rows'], 1)
        self.assertEqual(report['audit']['corrected_labels'], 3)
        families = [{family_key(chess.Board(e.fen)) for e in datasets[s]} for s in SPLITS]
        self.assertFalse(families[0] & families[1] or families[0] & families[2] or families[1] & families[2])

    def test_generation_is_reproducible_and_manifest_hashes_match(self):
        with TemporaryDirectory() as directory:
            source = self.source_for_all_splits(directory)
            first, report = self.collect(source)
            second, _ = self.collect(source)
            self.assertEqual(first, second)
            output = Path(directory) / 'output'
            write_datasets(output, first, report)
            manifest = json.loads((output / 'manifest.json').read_text())
            for split in SPLITS:
                content = (output / f'{split}.txt').read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), manifest['outputs'][f'{split}.txt']['sha256'])

    def test_insufficient_diversity_fails_instead_of_repeating_examples(self):
        with TemporaryDirectory() as directory:
            source = self.source_for_all_splits(directory)
            with self.assertRaisesRegex(ValueError, 'Insufficient source diversity'):
                self.collect(source, capacity=2)

    def test_existing_output_is_preserved(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'existing'
            output.mkdir()
            sentinel = output / 'keep.txt'
            sentinel.write_text('keep')
            with self.assertRaises(FileExistsError):
                write_datasets(output, {}, {})
            self.assertEqual(sentinel.read_text(), 'keep')

    def test_cli_rejects_bad_fractions_and_missing_files_with_84(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(['--validation-fraction', 'nan'])
            self.assertEqual(error.exception.code, 84)
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            self.assertEqual(main(['--input', str(Path(directory) / 'missing.txt'),
                                   '--output', str(Path(directory) / 'new')]), 84)
