"""A/B: perte quadratique vs cross-entropy, compteur brut vs normalisé. Même init, mêmes données."""
import sys, random, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
import numpy as np
from analyzer.chess.chessfile import ChessFile
from analyzer.chess.chessencoder import ChessEncoder
from analyzer.neural_network import NeuralNetwork
from analyzer import training

variant, start_nn, train_path, val_path, n_train, epochs, lr = sys.argv[1:8]
n_train, epochs, lr = int(n_train), int(epochs), float(lr)
loss_kind, clock = variant.split('_')  # mse|ce , raw|norm

def load(path, n):
    pos = [p for p in ChessFile(path).fen_games if p.label][:n]
    net_labels = net.labels
    out = []
    for p in pos:
        x = ChessEncoder(p).encode()
        if clock == 'norm':
            x[-1] = min(x[-1], 100.0) / 100.0
        out.append((x, training._make_target(' '.join(p.label), net)))
    return out

net = NeuralNetwork.load_from_file(start_nn)
net.learning_rate = lr
train = load(train_path, n_train)
val = load(val_path, 10**9)

def ce_step(network, x, t):
    acts, pre = network.forward_pass(x, return_preactivations=True)
    out = acts[-1]
    loss = -float(np.log(max(out[np.argmax(t)], 1e-12)))
    delta = out - t  # softmax + cross-entropy
    for i in range(len(network.weights) - 1, -1, -1):
        dW = np.outer(delta, acts[i]); db = delta
        if i > 0:
            g = network.weights[i].T @ delta
            nxt = training._activation_backward(network.activation_for_layer(i - 1), pre[i - 1], g)
        network.weights[i] -= network.learning_rate * dW
        network.biases[i] -= network.learning_rate * db
        if i > 0:
            delta = nxt
    return loss

if loss_kind == 'ce':
    training._backprop_step = ce_step

def acc(data):
    return np.mean([np.argmax(net.forward_pass(x)[-1]) == np.argmax(t) for x, t in data])

rng = random.Random(42)
t0 = time.time()
for e in range(1, epochs + 1):
    training.train_network(net, train, e, rng=rng)
    print(f"{variant} lr={lr} epoch {e:2d} val_acc={acc(val)*100:.2f}% t={time.time()-t0:.0f}s", flush=True)
