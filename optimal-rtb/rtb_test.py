#!/usr/bin/env python3
import sys
import random
import numpy as np
from ortb import tune_lambda_values_on_historical_auction, fit_parameters_for_bid_win_curve, objective

random.seed(10)

BUDGET_PROPORTIONS = [64, 16]  # simulated budget = total_real_cost / proportion

# Parameter grids to sweep for each strategy (kept identical to the original)
CONST_PARAS = list(range(2, 20, 2)) + list(range(20, 100, 5)) + list(range(100, 301, 10))
RAND_PARAS = list(range(2, 20, 2)) + list(range(20, 100, 5)) + list(range(100, 501, 10))
MCPC_PARAS = [1]
LIN_PARAS = (list(range(2, 20, 2)) + list(range(20, 100, 5)) +
             list(range(100, 400, 10)) + list(range(400, 800, 50)))
ORTB_PARAS = [10e-6]

ALGO_PARAS = {
    'const': CONST_PARAS,
    'rand': RAND_PARAS,
    'mcpc': MCPC_PARAS,
    'lin': LIN_PARAS,
    'ortb':ORTB_PARAS
}


# ---- bidding strategies ----
def bid_const(price):
    return price


def bid_rand(upper):
    return int(random.random() * upper)


def bid_mcpc(ecpc, pctr):
    return int(ecpc * pctr)

def bid_lin(pctr, base_ctr, base_bid):
    return int(pctr * base_bid / base_ctr)

def bid_ortb(pctr, lam, c):
    return int(np.sqrt(c**2 + (c*pctr)/lam) - c)


def win_auction(click_and_price, bid):
    """We win if our bid beats the real recorded market price."""
    _click, market_price = click_and_price
    return bid > market_price


def compute_bid(algo, para, pctr, original_ecpc, original_ctr, lam, c):
    if algo == 'const':
        return bid_const(para)
    elif algo == 'rand':
        return bid_rand(para)
    elif algo == 'mcpc':
        return bid_mcpc(original_ecpc, pctr)
    elif algo == 'lin':
        return bid_lin(pctr, original_ctr, para)
    elif algo == 'ortb':
        return bid_ortb(pctr, lam, c)
    else:
        raise ValueError(f'wrong bidding strategy name: {algo}')


def simulate_one_setting(cases, pctrs, total_cost, proportion, algo, para, original_ecpc, original_ctr, lam, c):
    """
    Replay the whole test log once for one (proportion, algo, para) setting.
    cases: list of (click, market_price) tuples, in original log order.
    pctrs: predicted CTR for each case, same order/length as cases.
    """
    budget = int(total_cost / proportion)
    cost = 0
    clicks = 0
    bids = 0
    imps = 0

    for idx, case in enumerate(cases):
        pctr = pctrs[idx]
        bid = compute_bid(algo, para, pctr, original_ecpc, original_ctr, lam, c)
        bids += 1
        if win_auction(case, bid):
            imps += 1
            clicks += case[0]
            cost += case[1]
        if cost > budget:
            break

    return (proportion, clicks, bids, imps, budget, cost, algo, para)


def format_result_row(result):
    proportion, clicks, bids, impressions, budget, cost, algorithm, para = result
    return '\t'.join(str(x) for x in (proportion, clicks, bids, impressions, budget, cost, algorithm, para))


def compute_original_stats(train_yzx_path):
    """
    Read train.yzx.txt (no header -- every line is data) to get:
      original_ctr  = overall click-through rate in training data
      original_ecpc = overall effective cost-per-click in training data
    These are used as reference points by the mcpc and lin strategies.
    """
    total_clicks = 0.0
    total_cost = 0.0
    num_impressions = 0
    with open(train_yzx_path, 'r') as fi:
        for line in fi:
            fields = line.split(' ')
            click = int(fields[0])
            cost = int(fields[1])
            num_impressions += 1
            total_clicks += click
            total_cost += cost
    original_ecpc = total_cost / total_clicks
    original_ctr = total_clicks / num_impressions
    return original_ecpc, original_ctr, total_cost


def load_test_cases(test_yzx_path):
    """Read test.yzx.txt (no header) -> list of (click, market_price), and total real cost."""
    cases = []
    total_cost = 0
    with open(test_yzx_path, 'r') as fi:
        for line in fi:
            fields = line.split(' ')
            click = int(fields[0])
            market_price = int(fields[1])
            cases.append((click, market_price))
            total_cost += market_price
    return cases, total_cost


def load_pctrs(pred_path):
    with open(pred_path, 'r') as fi:
        return [float(line.strip()) for line in fi]


def main():
    if len(sys.argv) < 7:
        print('Usage: python3 rtb_test.py train.log.txt train.yzx.txt test.yzx.txt train.yzx.txt.lr.weight test.yzx.txt.lr.pred rtb.results.tsv')
        sys.exit(-1)

    train_log_path, train_yzx_path, test_yzx_path, feat_weights_path, pred_path, results_path = sys.argv[1:7]

    original_ecpc, original_ctr, train_total_cost = compute_original_stats(train_yzx_path)
    cases, total_cost = load_test_cases(test_yzx_path)
    pctrs = load_pctrs(pred_path)

    lambda_list = [10**x for x in range(-10, 10)]
    c = fit_parameters_for_bid_win_curve(objective, train_log_path)

    header = 'prop\tclks\tbids\timps\tbudget\tspend\talgo\tpara'
    print(header)
    with open(results_path, 'w') as fo:
        fo.write(header + '\n')
        for proportion in BUDGET_PROPORTIONS:
            lam = tune_lambda_values_on_historical_auction(
                train_total_cost/proportion,train_yzx_path, feat_weights_path, lambda_list, c
            )
            ALGO_PARAS['ortb'] = [lam]
            for algo, paras in ALGO_PARAS.items():
                for para in paras:
                    result = simulate_one_setting(
                        cases, pctrs, total_cost, proportion, algo, para,
                        original_ecpc, original_ctr,lam,c
                    )
                    row = format_result_row(result)
                    print(row)
                    fo.write(row + '\n')


if __name__ == '__main__':
    main()
