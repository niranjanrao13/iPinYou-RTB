import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from lr import parse_yzx_line, sigmoid
from scipy.optimize import curve_fit
from collections import defaultdict



# Define parametric form of bid-win curve
def objective(x,c):
  return x/(c+x)

def objective2(x,c):
  return (x**2)/(c**2 + x**2)

def ortb1_bid(c, theta, lambda_):
    return np.sqrt(c**2 + (c*theta)/lambda_) - c

def read_feat_weights(feat_weights_path):
    wts = {}
    with open(feat_weights_path,'r') as f:
        feat_weights = f.readlines()

    for line in feat_weights:
        feat_index,feat_wt = line.split('\t')
        wts[int(feat_index)] = float(feat_wt)

    return wts
    
# Parametric curve fitting
def fit_parameters_for_bid_win_curve(objective_fn, train_log_path):
    train_log = pd.read_csv(train_log_path,sep='\t')
    bid_prices = np.arange(0,300,10)
    check_win = lambda x, bid_ : 1 if x >= bid_ else 0
    w_values = [sum(check_win(bid_price,val) for val in train_log['payprice'].values) for bid_price in bid_prices]
    w_probs = [x/len(train_log['payprice'].values) for x in w_values]
    c_param,_ = curve_fit(objective_fn,bid_prices,w_probs)
    # plt.plot(bid_prices, w_probs)
    return c_param.item()

# Get best value of lambda by searching through log space
def tune_lambda_values_on_historical_auction(budget, train_yzx_path, feat_weights_path,lambda_list,c_param):
    with open(train_yzx_path,'r') as f:
        train_data = f.readlines()

    feat_weights = read_feat_weights(feat_weights_path)
    
    number_of_bids_won_vs_lambda = defaultdict(int)
    spends_vs_lambda = defaultdict(int)
    clicks_vs_lambda = defaultdict(int)
    impressions_vs_lambda = defaultdict(int)

    # auction replay
    for line in train_data:
        data = parse_yzx_line(line)
        click_yes_no = data[0] 
        payprice = data[1]
        net_score = sum(feat_weights[x] for x in data[2:])
        pctr = sigmoid(net_score)
        
        for lambda_ in lambda_list:
            bid = ortb1_bid(c_param,pctr,lambda_)
            total_spend = spends_vs_lambda[lambda_]

            if total_spend > budget:
                continue

            if bid >= payprice: 
                number_of_bids_won_vs_lambda[lambda_] += 1
                spends_vs_lambda[lambda_] += bid
                impressions_vs_lambda[lambda_] += 1
                clicks_vs_lambda[lambda_] += click_yes_no


    return max(clicks_vs_lambda, key=clicks_vs_lambda.get)



def main():
    from rtb_test import compute_original_stats  # deferred, see note above

    if len(sys.argv) < 4:
        print('Usage: python ortb.py train.log.txt train.yzx.txt train.yzx.txt.lr.weight')
        exit(-1)
        
    train_log_path = sys.argv[1]
    train_yzx_path = sys.argv[2]
    train_feat_weights_path = sys.argv[3]
    lambda_logspace = [10**(x) for x in range(-10,10)]

    
    c_param = fit_parameters_for_bid_win_curve(objective, train_log_path)
    _,_,budget = compute_original_stats(train_yzx_path)
    best_lambda = tune_lambda_values_on_historical_auction(budget, train_yzx_path, train_feat_weights_path, lambda_logspace, c_param)
    
    print(c_param)
    print(best_lambda)


if __name__ == '__main__':
   main()