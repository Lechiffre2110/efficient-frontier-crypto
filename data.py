import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as sco
import plotly
import plotly.graph_objects as go
from flask import Flask, make_response, jsonify, request
from flask_cors import CORS

RISK_FREE_RATE_SYMBOL = '^TNX' #10 year treasury yield
#RISK_FREE_RATE = yf.Ticker(RISK_FREE_RATE_SYMBOL).history(period="1d").iloc[0]['Close']/100
RISK_FREE_RATE = 0.03

timeframe = 365 #TODO: rename and find a way to dynamically adjust this based on the asset type


def getAssetData(asset, start, end): 
    """Returns the mean returns and covariance matrix of the asset data from yahoo finance in the specified time period"""
    data = yf.download(asset, start, end)['Close']
    daily_returns = data.pct_change()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    return mean_returns, cov_matrix

def calculateTotalPortfolioPerformance(assetWeights, mean_returns, cov_matrix):
    """Returns the total portfolio performance in terms of returns and standard deviation"""
    returns = np.sum(mean_returns*assetWeights) * timeframe #250 for traditional assets & 365 for crypto assets
    std = np.sqrt(np.dot(assetWeights.T, np.dot(cov_matrix, assetWeights))) * np.sqrt(timeframe) #250 for traditional assets & 365 for crypto assets
    return returns, std


def calculateSharpeRatio(assetWeights, mean_returns, cov_matrix, risk_free_rate = RISK_FREE_RATE):
    """Returns the sharpe ratio of the portfolio given the asset weights, mean returns and covariance matrix"""
    pReturns, pStd = calculateTotalPortfolioPerformance(assetWeights, mean_returns, cov_matrix)
    return -((pReturns - risk_free_rate) / pStd) 

def maxSharpeRatio(mean_returns, cov_matrix, constraint_set, risk_free_rate = RISK_FREE_RATE): 
    """Returns the maximum sharpe ratio (by minimizing the neg. sharpe ratio) and the optimal asset weights given the mean returns and covariance matrix"""
    number_of_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #check if the sum of weights is 1
    bounds = tuple(constraint_set for asset in range(number_of_assets))
    result = sco.minimize(calculateSharpeRatio, number_of_assets*[1./number_of_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def calculatePortfolioVariance(assetWeights, mean_returns, cov_matrix):
    """Returns the portfolio variance given the asset weights, mean returns and covariance matrix"""
    return calculateTotalPortfolioPerformance(assetWeights, mean_returns, cov_matrix)[1]

def minVolatility(mean_returns, cov_matrix, constraint_set):
    """Returns the minimum volatility portfolio (by minimizing the portfolio variance) and the optimal asset weights given the mean returns and covariance matrix"""

    number_of_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for asset in range(number_of_assets))
    result = sco.minimize(calculatePortfolioVariance, number_of_assets*[1./number_of_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def portfolio_returns(asset_weights, mean_returns, cov_matrix):
    """Returns the portfolio returns given the asset weights, mean returns and covariance matrix"""
    return calculateTotalPortfolioPerformance(asset_weights, mean_returns, cov_matrix)[0]

def efficient_frontier(mean_returns, cov_matrix, target_return, constraint_set):
    """For a given set of target returns, returns the optimal asset weights for each target return"""

    asset_number = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_returns(x, mean_returns, cov_matrix) - target_return}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) 
    bounds = tuple(constraint_set for asset in range(asset_number))
    efficient_frontier_result = sco.minimize(calculatePortfolioVariance, asset_number*[1./asset_number,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    return efficient_frontier_result

def getMaxSharpeRatioPortfolio(mean_returns, cov_matrix, constraint_set, risk_free_rate = RISK_FREE_RATE):
    '''Returns the maximum sharpe ratio portfolio given the mean returns and covariance matrix'''

    max_sharpe_ratio_portfolio = maxSharpeRatio(mean_returns, cov_matrix, constraint_set) #INCLUDE CONSTRAINTS
    max_sharpe_ratio_returns, max_sharpe_ratio_std = calculateTotalPortfolioPerformance(max_sharpe_ratio_portfolio['x'], mean_returns, cov_matrix)
    max_sharpe_ratio_weights = pd.DataFrame(max_sharpe_ratio_portfolio['x'], index=mean_returns.index, columns=['weight'])
    max_sharpe_ratio_weights.weight = [round(i*100,2)for i in max_sharpe_ratio_weights.weight]
    return max_sharpe_ratio_returns, max_sharpe_ratio_std, max_sharpe_ratio_weights

def getMinVolatilityPortfolio(mean_returns, cov_matrix, constraint_set, risk_free_rate = RISK_FREE_RATE):
    min_volatility_portfolio = minVolatility(mean_returns, cov_matrix, constraint_set) #INCLUDE CONSTRAINTS
    min_volatility_returns, min_volatility_std = calculateTotalPortfolioPerformance(min_volatility_portfolio['x'], mean_returns, cov_matrix)
    min_volatility_weights = pd.DataFrame(min_volatility_portfolio['x'], index=mean_returns.index, columns=['weight'])
    min_volatility_weights.weight = [round(i*100,2)for i in min_volatility_weights.weight]
    return min_volatility_returns, min_volatility_std, min_volatility_weights

def getEfficientFrontier(mean_returns, cov_matrix, constraint_set):
    max_sharpe_ratio_returns, max_sharpe_ratio_std, max_sharpe_ratio_weights = getMaxSharpeRatioPortfolio(mean_returns, cov_matrix, constraint_set)
    min_volatility_returns, min_volatility_std, min_volatility_weights = getMinVolatilityPortfolio(mean_returns, cov_matrix, constraint_set)
    target_returns = np.linspace(min_volatility_returns, max_sharpe_ratio_returns, 100)
    efficient_frontier_list = []
    efficient_frontier_weight_list = []

    for target in target_returns:
        result = efficient_frontier(mean_returns, cov_matrix, target, constraint_set)
        efficient_frontier_list.append((result['fun']))
        efficient_frontier_weight_list.append(result['x'])
    
    return max_sharpe_ratio_returns, max_sharpe_ratio_std, max_sharpe_ratio_weights, min_volatility_returns, min_volatility_std, min_volatility_weights, target_returns, efficient_frontier_list, efficient_frontier_weight_list

'''
#TODO: remove this function, as it is replaced by the function in API function, alternatively move the code from the endpoint to this function
def plotEfficientFrontier(mean_returns, cov_matrix, constraint_set=(0, 1)):
    max_sharpe_ratio_returns, max_sharpe_ratio_std, max_sharpe_ratio_weights, min_volatility_returns, min_volatility_std, min_volatility_weights, target_returns, efficient_frontier, efficient_frontier_weight_list = getEfficientFrontier(mean_returns, cov_matrix)

    # Create scatter plots for Maximum Sharpe Ratio and Minimum Volatility points
    max_sharpe = go.Scatter(x=[max_sharpe_ratio_std], y=[max_sharpe_ratio_returns], mode='markers', name='Maximum Sharpe Ratio', marker=dict(color='red', size=10))
    min_volatility = go.Scatter(x=[min_volatility_std], y=[min_volatility_returns], mode='markers', name='Minimum Volatility', marker=dict(color='blue', size=10))

    # Create line plot for the Efficient Frontier
    efficient_frontier_line = go.Scatter(x=efficient_frontier, y=target_returns, mode='lines', name='Efficient Frontier', line=dict(dash='dash', color='black'))

    layout = go.Layout(
        title='Efficient Frontier',
        xaxis=dict(title='Volatility'),
        yaxis=dict(title='Returns'),
        hovermode='closest'
    )


    data = [max_sharpe, min_volatility, efficient_frontier_line]
    fig = go.Figure(data=data, layout=layout)
    fig.show()
    '''


# Flask Endpoint

endtime = dt.datetime.now()

app = Flask(__name__)
CORS(app)

@app.route('/portfolio')
def returnEfficientFrontier():
    global timeframe
    constraints = (0,1)
    if (request.args.get('includeAllAssets') == 'true'):
        constraints = (0.02,1)

    assets = request.args.get('assets')
    assetNames = assets.split(',')
    days = int(request.args.get('days'))
    timeframe = days
    starttime = endtime - dt.timedelta(days=days)

    mean_returns, cov_matrix = getAssetData(assetNames, starttime, endtime)
    max_sharpe_ratio_returns, max_sharpe_ratio_std, max_sharpe_ratio_weights, min_volatility_returns, min_volatility_std, min_volatility_weights, target_returns, efficient_frontier, efficient_frontier_weight_list = getEfficientFrontier(mean_returns, cov_matrix, constraints)

    columns = ['Portfolio', 'Returns (%)', 'Volatility (%)',] + [asset for asset in assetNames]
    rows = []

    portfolio_data_frame = pd.DataFrame(columns=columns)
    max_sharpe_ratio_portfolio_data = ['Maximum Sharpe Ratio', round(max_sharpe_ratio_returns*100,2), round(max_sharpe_ratio_std*100,2)] + max_sharpe_ratio_weights['weight'].tolist()
    min_volatility_portfolio_data = ['Minimum Volatility', round(min_volatility_returns*100,2), round(min_volatility_std*100,2)] + min_volatility_weights['weight'].tolist()
    rows.append(max_sharpe_ratio_portfolio_data)
    rows.append(min_volatility_portfolio_data)

    allPortfolios = pd.DataFrame(np.round(efficient_frontier_weight_list,2), columns=assetNames)
    allPortfolios['Returns'] = np.round(target_returns*100,2)
    allPortfolios['Volatility'] = [round(val * 100, 2) for val in efficient_frontier]
    allPortfolios['Sharpe Ratio'] = ((allPortfolios['Returns'] - RISK_FREE_RATE) / allPortfolios['Volatility']).round(2)
    allPortfolios = allPortfolios[['Sharpe Ratio', 'Returns', 'Volatility'] + assetNames]

    # Assign the data rows to the DataFrame
    portfolio_data_frame = pd.DataFrame(rows, columns=columns)


    # Create scatter plots for Maximum Sharpe Ratio and Minimum Volatility points
    max_sharpe = go.Scatter(x=[max_sharpe_ratio_std], y=[max_sharpe_ratio_returns], mode='markers', name='Maximum Sharpe Ratio', marker=dict(color='red', size=10))
    min_volatility = go.Scatter(x=[min_volatility_std], y=[min_volatility_returns], mode='markers', name='Minimum Volatility', marker=dict(color='blue', size=10))

    # Create line plot for the Efficient Frontier
    efficient_frontier_line = go.Scatter(x=efficient_frontier, y=target_returns, mode='lines', name='Efficient Frontier', line=dict(dash='dash', color='black'))

    data = [max_sharpe, min_volatility, efficient_frontier_line]
    layout = go.Layout(
        title='Efficient Frontier',
        xaxis=dict(title='Volatility'),
        yaxis=dict(title='Expected Returns'),
        hovermode='closest'
    )
    fig = go.Figure(data=data, layout=layout)

    response = {
        'efficientFrontier': plotly.io.to_json(fig, pretty=True),
        'portfolio': portfolio_data_frame.to_json(orient='records'),
        'allPortfolios': allPortfolios.to_json(orient='records')
    }

    return response

#@app.route('/portfolio/custom')
#def returnEfficientFrontierCustomPortfolio():



if __name__ == '__main__':
    app.run(debug=True)