"""Génère des positions inédites (absentes de all.txt) via des parties semi-aléatoires."""
import random, sys
import chess

all_path, out_path, per_class = sys.argv[1], sys.argv[2], int(sys.argv[3])
known = set()
for line in open(all_path):
    f = line.split()
    known.add(' '.join(f[:4]))

def label(b):
    side = 'Black' if b.turn == chess.WHITE else 'White'
    if b.is_checkmate(): return f'Checkmate {side}'
    if b.is_check(): return f'Check {side}'
    return 'Nothing'

rng = random.Random(2026)
buckets = {k: [] for k in ('Checkmate White', 'Checkmate Black', 'Check White', 'Check Black', 'Nothing')}
seen = set()
games = 0
while min(len(v) for v in buckets.values()) < per_class and games < 400000:
    games += 1
    b = chess.Board()
    for _ in range(rng.randint(10, 160)):
        moves = list(b.legal_moves)
        if not moves: break
        r = rng.random()
        mates = [m for m in moves if b.gives_check(m) and (b.push(m) or True) and (b.is_checkmate(), b.pop())[0]] if r < 0.5 else []
        checks = [m for m in moves if b.gives_check(m)] if r < 0.3 else []
        caps = [m for m in moves if b.is_capture(m)] if r < 0.6 else []
        b.push(rng.choice(mates or checks or caps or moves))
        key = ' '.join(b.fen().split()[:4])
        lab = label(b)
        if key in known or key in seen or len(buckets[lab]) >= per_class: continue
        if lab == 'Nothing' and rng.random() > 0.05: continue
        seen.add(key)
        buckets[lab].append(f'{b.fen()} {lab}')
        if b.is_game_over(): break
lines = [l for v in buckets.values() for l in v]
rng.shuffle(lines)
open(out_path, 'w').write('\n'.join(lines) + '\n')
print(games, {k: len(v) for k, v in buckets.items()})
