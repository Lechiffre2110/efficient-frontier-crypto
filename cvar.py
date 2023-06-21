import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.stats as scs
import plotly
import plotly.graph_objects as go
from flask import Flask, make_response, jsonify, request
from flask_cors import CORS

RISK_FREE_RATE_SYMBOL = '^TNX' #10 year treasury yield
#RISK_FREE_RATE = yf.Ticker(RISK_FREE_RATE_SYMBOL).history(period="1d").iloc[0]['Close']/100
RISK_FREE_RATE = 0.03
ALPHA = 0.05

timeframe = 365 #TODO: rename and find a way to dynamically adjust this based on the asset type


def getAssetData(asset, start, end): 
    """Returns the mean returns and covariance matrix of the asset data from yahoo finance in the specified time period"""
    data = yf.download(asset, start, end)['Close']
    daily_returns = data.pct_change()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    return mean_returns, cov_matrix

def calculateCVaR(assetWeights, mean_returns, cov_matrix, alpha=ALPHA):
    """Returns the cVaR of the portfolio given the asset weights, mean returns, covariance matrix, and significance level alpha"""
    portfolio_returns = np.sum(mean_returns * assetWeights)
    portfolio_volatility = np.sqrt(np.dot(assetWeights.T, np.dot(cov_matrix, assetWeights)))
    z_score = np.abs(scs.norm.ppf(alpha))
    cvar = z_score * portfolio_volatility - portfolio_returns
    return cvar

def efficient_frontier_cvar(mean_returns, cov_matrix, target_cvar, constraint_set):
    """For a given set of target cVaR, returns the optimal asset weights for each target cVaR"""

    asset_number = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = [{'type': 'eq', 'fun': lambda x: calculateCVaR(x, mean_returns, cov_matrix) - target_cvar},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple(constraint_set for asset in range(asset_number))
    efficient_frontier_result = sco.minimize(calculateCVaR, asset_number * [1. / asset_number, ], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return efficient_frontier_result

def getEfficientFrontierCVaR(mean_returns, cov_matrix, constraint_set):
    """For a given set of target CVaR, returns the optimal asset weights for each target CVaR"""
    target_cvars = np.linspace(0, 0.2, 100)  # Example range from 0 to 0.2 (modify as needed)
    efficient_frontier_weight_list = []

    for target in target_cvars:
        result = efficient_frontier_cvar(mean_returns, cov_matrix, target, constraint_set)
        efficient_frontier_weight_list.append(result['x'])
    return target_cvars, efficient_frontier_weight_list



# Flask Endpoint

endtime = dt.datetime.now()

app = Flask(__name__)
CORS(app)

@app.route('/portfolio')
def returnEfficientFrontier():
    global timeframe
    constraints = (0,1)

    #if (request.args.get('includeAllAssets') == 'true'):
     #   constraints = (0.02,1)

    assets = request.args.get('assets')
    assetNames = assets.split(',')
    days = int(request.args.get('days'))
    timeframe = days
    starttime = endtime - dt.timedelta(days=days)

    mean_returns, cov_matrix = getAssetData(assetNames, starttime, endtime)

    target_cvars, efficient_frontier_weight_list = getEfficientFrontierCVaR(mean_returns, cov_matrix, constraints)

    allPortfolios = pd.DataFrame(np.round(efficient_frontier_weight_list, 2), columns=assetNames)
    allPortfolios['CVaR'] = target_cvars.round(4)

    #plot efficient frontier
    efficient_frontier_line = go.Scatter(x=allPortfolios['CVaR'], y=target_cvars, mode='lines', name='Efficient Frontier', line=dict(dash='dash', color='black'))

    data = [efficient_frontier_line]
    layout = go.Layout(
        title='Efficient Frontier (cVaR)',
        xaxis=dict(title='cVaR'),
        yaxis=dict(title='Return'),
        hovermode='closest'
    )
    fig = go.Figure(data=data, layout=layout
    


    print(allPortfolios)

    response = {
        'allPortfolios': allPortfolios.to_json(orient='records')
    }

    return response
    """
    response = {
            'efficientFrontier': plotly.io.to_json(fig, pretty=True),
            'portfolio': portfolio_data_frame.to_json(orient='records'),
            'allPortfolios': allPortfolios.to_json(orient='records')
        }

        return response
    """

"""

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
        yaxis=dict(title='Returns'),
        hovermode='closest'
    )
    fig = go.Figure(data=data, layout=layout)

    response = {
        'efficientFrontier': plotly.io.to_json(fig, pretty=True),
        'portfolio': portfolio_data_frame.to_json(orient='records'),
        'allPortfolios': allPortfolios.to_json(orient='records')
    }

    return response
    """




if __name__ == '__main__':
    app.run(debug=True)