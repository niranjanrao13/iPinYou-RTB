#!/usr/bin/env python3
"""
lr.py  --  Online logistic-regression CTR estimator.

This is a clean, commented Python 3 port of the original optimal-rtb
`lryzx.py`. It trains directly on the sparse "yzx" feature files produced
by make-ipinyou-data:

    click market_price feat_id1:1 feat_id2:1 feat_id3:1 ...

- "click"        is the label y (0 or 1)
- "market_price" is z, the winning price of the auction (unused for training,
                  carried through for the bidding simulation later)
- each "feat_idN:1" is a one-hot feature that's present for this impression
  (e.g. "this impression's city was Beijing", "this slot was 300x250", ...)

MODEL
-----
Because every present feature has value 1 (one-hot encoding), the linear
score for an impression is just the SUM of the weights of its active
features:

    score = sum(w[feat] for feat in active_features)
    pCTR  = sigmoid(score)

TRAINING
--------
Plain online stochastic gradient descent, one impression at a time, with a
small L2-style weight decay, repeated for a fixed number of passes
("rounds") over the training file. For each impression:

    pred = sigmoid(sum of weights of its active features)
    for each active feature f:
        w[f] = w[f] * (1 - lamb) + eta * (click - pred)

This matches the *behavior* of the original code. Note this isn't quite the
textbook regularized-SGD update (that would decay by (1 - eta*lamb), not
(1 - lamb)) -- but with lamb=1e-6 the difference is negligible, and this is
kept as-is to match the original benchmark's published numbers.

Usage:
    python3 lr.py <path_to_>/train.yzx.txt <path_to_>/test.yzx.txt

Outputs (next to the input files):
    train.yzx.txt.lr.weight   -- learned feature weights
    test.yzx.txt.lr.pred      -- predicted pCTR, one per line, for test.yzx.txt
"""
import sys
import math
import random
import operator
from sklearn.metrics import roc_auc_score, mean_squared_error

# ---- hyperparameters (unchanged from the original benchmark) ----
BUFFER_CASE_NUM = 1_000_000   # how many training lines to batch before updating
LEARNING_RATE = 0.01          # eta
L2_DECAY = 1e-6               # lamb
TRAIN_ROUNDS = 10             # passes over the training data
INIT_WEIGHT_SCALE = 0.05      # new feature weights start in [-0.025, 0.025]

FEATURE_START_COL = 2         # columns 0,1 are click,market_price; features start at 2

random.seed(10)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def next_init_weight():
    return (random.random() - 0.5) * INIT_WEIGHT_SCALE


def parse_yzx_line(line):
    """'1 106 3:1 891:1 ...' -> [1, 106, 3, 891, ...] (click, price, feat_ids...)"""
    return [int(tok) for tok in line.replace(":1", "").split()]


def predict(feat_weights, feat_ids):
    """Sum the weights of active features (creating any new ones at 0-init), then sigmoid."""
    score = 0.0
    for feat in feat_ids:
        if feat not in feat_weights:
            feat_weights[feat] = next_init_weight()
        score += feat_weights[feat]
    return sigmoid(score)


def sgd_update(feat_weights, feat_ids, click, pred):
    for feat in feat_ids:
        feat_weights[feat] = feat_weights[feat] * (1 - L2_DECAY) + LEARNING_RATE * (click - pred)


def train_one_pass(train_path, feat_weights):
    """One full pass over the training file, updating feat_weights in place (mini-batched)."""
    buffer = []
    with open(train_path, 'r') as fi:
        for line in fi:
            buffer.append(parse_yzx_line(line))
            if len(buffer) >= BUFFER_CASE_NUM:
                _consume_buffer(buffer, feat_weights)
                buffer = []
        if buffer:
            _consume_buffer(buffer, feat_weights)


def _consume_buffer(buffer, feat_weights):
    for data in buffer:
        click, _market_price = data[0], data[1]
        feat_ids = data[FEATURE_START_COL:]
        pred = predict(feat_weights, feat_ids)
        sgd_update(feat_weights, feat_ids, click, pred)


def evaluate(test_path, feat_weights):
    """Return (auc, rmse) of the current model on the test file."""
    y_true, y_pred = [], []
    with open(test_path, 'r') as fi:
        for line in fi:
            data = parse_yzx_line(line)
            click = data[0]
            feat_ids = data[FEATURE_START_COL:]
            # NOTE: at eval time we do NOT create new weights for unseen features --
            # unseen features simply contribute 0 to the score.
            score = sum(feat_weights[f] for f in feat_ids if f in feat_weights)
            y_true.append(click)
            y_pred.append(sigmoid(score))
    auc = roc_auc_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return auc, rmse, y_pred


def save_weights(path, feat_weights):
    with open(path, 'w') as fo:
        for feat, weight in sorted(feat_weights.items(), key=operator.itemgetter(0)):
            fo.write(f'{feat}\t{weight}\n')


def save_predictions(path, y_pred):
    with open(path, 'w') as fo:
        for p in y_pred:
            fo.write(f'{p}\n')


def main():
    if len(sys.argv) < 3:
        print('Usage: python3 lr.py <path-to-train>.yzx.txt <path-to-test>.yzx.txt')
        sys.exit(-1)

    train_path, test_path = sys.argv[1], sys.argv[2]
    feat_weights = {}

    print('round\tauc\trmse')
    for round_idx in range(TRAIN_ROUNDS):
        train_one_pass(train_path, feat_weights)
        auc, rmse, y_pred = evaluate(test_path, feat_weights)
        print(f'{round_idx}\t{auc}\t{rmse}')

    save_weights(train_path + '.lr.weight', feat_weights)
    save_predictions(test_path + '.lr.pred', y_pred)  # from the final round's evaluation


if __name__ == '__main__':
    main()
