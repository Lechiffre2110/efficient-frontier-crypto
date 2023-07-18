import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import scipy.optimize as sco
import plotly
import plotly.graph_objects as go
from flask import Flask, request
from flask_cors import CORS

__author__ = "Younes Abdelwadoud"
__version__ = "0.0.1"
__license__ = "MIT"

def getRiskFreeRate():
    """
    Returns the risk free rate (10 year US treasury yield) from yahoo finance

    Returns:
    risk_free_rate (float): the risk free rate (10 year US treasury yield)

    Exception: if an error occurs while fetching the risk-free rate, a default value of 0.04 is returned
    """
    risk_free_rate_symbol = '^TNX'  # 10-year Treasury yield symbol
    
    try:
        tnx_data = yf.Ticker(risk_free_rate_symbol).history(period="1d")
        risk_free_rate = tnx_data.iloc[0]['Close'] / 100.0
        return risk_free_rate
    
    except Exception as e:
        print("Error occurred while fetching risk-free rate. Defaulting to a fixed rate.")
        return 0.04  # Default value in case of error


RISK_FREE_RATE = getRiskFreeRate()

# Default timeframe setup
timeframe = 365
endtime = dt.datetime.now()

def isCrypto(asset):
    """
    Returns true if the asset is a cryptocurrency, false if it isn't

    Args: 
    asset (string): the asset symbol (e.g. BTC-USD)

    Returns:
    bool: true if the asset is a cryptocurrency, false if it isn't
    """
    return asset[-3:] == "-USD"


def getAssetData(asset, start, end): 
    """
    Returns the mean returns and covariance matrix of the asset data from yahoo finance in the specified time period
    
    Args:
    asset (string): the asset symbol (e.g. BTC-USD)
    start (datetime): the start date of the time period
    end (datetime): the end date of the time period

    Returns:
    mean_returns (pandas.Series): the mean returns of the asset
    cov_matrix (pandas.DataFrame): the covariance matrix of the asset
    """
    data = yf.download(asset, start, end)['Close']
    daily_returns = data.pct_change()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    return mean_returns, cov_matrix

def calculateTotalPortfolioPerformance(assetWeights, mean_returns, cov_matrix):
    """
    Returns the total portfolio performance in terms of returns and standard deviation
    
    Args:
    assetWeights (list): the weights of each asset in the portfolio
    mean_returns (pandas.Series): the mean returns of each asset in the portfolio
    cov_matrix (pandas.DataFrame): the covariance matrix of the assets in the portfolio

    Returns:
    returns (float): the total portfolio returns
    std (float): the total portfolio standard deviation
    """
    returns = np.sum(mean_returns * assetWeights) * timeframe
    std = np.sqrt(np.dot(assetWeights.T, np.dot(cov_matrix, assetWeights))) * np.sqrt(timeframe)
    
    if isCrypto(mean_returns.index[0]) == False:
        returns = returns * 252 / 365
    
    return returns, std

def calculateSharpeRatio(assetWeights, mean_returns, cov_matrix, risk_free_rate = RISK_FREE_RATE):
    """
    Returns the sharpe ratio of the portfolio given the asset weights, mean returns and covariance matrix
    
    Args:
    assetWeights (list): the weights of each asset in the portfolio
    mean_returns (pandas.Series): the mean returns of each asset in the portfolio
    cov_matrix (pandas.DataFrame): the covariance matrix of the assets in the portfolio
    risk_free_rate (float): 10 year US treasury yield

    Returns:
    sharpe_ratio (float): the sharpe ratio of the portfolio
    """
    pReturns, pStd = calculateTotalPortfolioPerformance(assetWeights, mean_returns, cov_matrix)
    return -((pReturns - risk_free_rate) / pStd) 

def maxSharpeRatio(mean_returns, cov_matrix, constraint_set, risk_free_rate = RISK_FREE_RATE): 
    """
    Returns the maximum sharpe ratio (by minimizing the neg. sharpe ratio) and the optimal asset weights given the mean returns and covariance matrix
    
    Args:
    mean_returns (pandas.Series): the mean returns of each asset in the portfolio
    cov_matrix (pandas.DataFrame): the covariance matrix of the assets in the portfolio
    constraint_set (list): the constraint set for the asset weights
    risk_free_rate (float): 10 year US treasury yield

    Returns:
    result (scipy.optimize.OptimizeResult): the result of the optimization
    """

    number_of_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    # Constraint to ensure that the sum of the weights is equal to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for asset in range(number_of_assets))

    # Minimize the negative sharpe ratio to maximize the sharpe ratio, by using the SLSQP method
    result = sco.minimize(calculateSharpeRatio, number_of_assets*[1./number_of_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def calculatePortfolioVariance(assetWeights, mean_returns, cov_matrix):
    """
    Returns the portfolio variance given the asset weights, mean returns and covariance matrix
    
    Args:
    assetWeights (list): the weights of each asset in the portfolio
    mean_returns (pandas.Series): the mean returns of each asset in the portfolio
    cov_matrix (pandas.DataFrame): the covariance matrix of the assets in the portfolio

    Returns:
    variance (float): the portfolio variance
    """
    return calculateTotalPortfolioPerformance(assetWeights, mean_returns, cov_matrix)[1]

def minVolatility(mean_returns, cov_matrix, constraint_set):
    """
    Returns the minimum volatility portfolio given the mean returns and covariance matrix

    Args:
    mean_returns (pandas.Series): the mean returns of each asset in the portfolio
    cov_matrix (pandas.DataFrame): the covariance matrix of the assets in the portfolio
    constraint_set (list): the constraint set for the asset weights

    Returns:
    result (scipy.optimize.OptimizeResult): the result of the optimization
    """

    number_of_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    # Constraint to ensure that the sum of the weights is equal to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for asset in range(number_of_assets))

    # Minimize the portfolio variance, by using the SLSQP method
    result = sco.minimize(calculatePortfolioVariance, number_of_assets*[1./number_of_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def portfolio_returns(asset_weights, mean_returns, cov_matrix):
    """
    Returns the portfolio returns given the asset weights, mean returns and covariance matrix
    
    Args:
    assetWeights (list): the weights of each asset in the portfolio
    mean_returns (pandas.Series): the mean returns of each asset in the portfolio
    cov_matrix (pandas.DataFrame): the covariance matrix of the assets in the portfolio

    Returns:
    returns (float): the portfolio returns
    """
    return calculateTotalPortfolioPerformance(asset_weights, mean_returns, cov_matrix)[0]

def efficient_frontier(mean_returns, cov_matrix, target_return, constraint_set):
    """
    For a given set of target returns, returns the optimal asset weights for each target return
    
    Args:
    mean_returns (pandas.Series): the mean returns of each asset in the portfolio
    cov_matrix (pandas.DataFrame): the covariance matrix of the assets in the portfolio
    target_return (list): the target returns for which the optimal asset weights are calculated
    constraint_set (list): the constraint set for the asset weights

    Returns:
    efficient_frontier_result (scipy.optimize.OptimizeResult): the result of the optimization
    """

    asset_number = len(mean_returns)
    args = (mean_returns, cov_matrix)

    # Constraint to ensure that the sum of the weights is equal to 1
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_returns(x, mean_returns, cov_matrix) - target_return}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) 
    bounds = tuple(constraint_set for asset in range(asset_number))

    # Minimize the portfolio variance, by using the SLSQP method
    efficient_frontier_result = sco.minimize(calculatePortfolioVariance, asset_number*[1./asset_number,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    return efficient_frontier_result

def getMaxSharpeRatioPortfolio(mean_returns, cov_matrix, constraint_set, risk_free_rate = RISK_FREE_RATE):
    """
    Returns the maximum sharpe ratio portfolio given the mean returns and covariance matrix
    
    Args:
    mean_returns (pandas.Series): the mean returns of each asset in the portfolio
    cov_matrix (pandas.DataFrame): the covariance matrix of the assets in the portfolio
    constraint_set (list): the constraint set for the asset weights
    risk_free_rate (float): 10 year US treasury yield

    Returns:
    max_sharpe_ratio_returns (float): the returns of the maximum sharpe ratio portfolio
    max_sharpe_ratio_std (float): the standard deviation of the maximum sharpe ratio portfolio
    max_sharpe_ratio_weights (pandas.DataFrame): the weights of the assets in the maximum sharpe ratio portfolio
    """

    max_sharpe_ratio_portfolio = maxSharpeRatio(mean_returns, cov_matrix, constraint_set) #INCLUDE CONSTRAINTS
    max_sharpe_ratio_returns, max_sharpe_ratio_std = calculateTotalPortfolioPerformance(max_sharpe_ratio_portfolio['x'], mean_returns, cov_matrix)
    max_sharpe_ratio_weights = pd.DataFrame(max_sharpe_ratio_portfolio['x'], index=mean_returns.index, columns=['weight'])
    max_sharpe_ratio_weights.weight = [round(i*100,2)for i in max_sharpe_ratio_weights.weight]
    return max_sharpe_ratio_returns, max_sharpe_ratio_std, max_sharpe_ratio_weights

def getMinVolatilityPortfolio(mean_returns, cov_matrix, constraint_set, risk_free_rate = RISK_FREE_RATE):
    """
    Returns the minimum volatility portfolio given the mean returns and covariance matrix
    
    Args:
    mean_returns (pandas.Series): the mean returns of each asset in the portfolio
    cov_matrix (pandas.DataFrame): the covariance matrix of the assets in the portfolio
    constraint_set (list): the constraint set for the asset weights
    risk_free_rate (float): 10 year US treasury yield

    Returns:
    min_volatility_returns (float): the returns of the minimum volatility portfolio
    min_volatility_std (float): the standard deviation of the minimum volatility portfolio
    min_volatility_weights (pandas.DataFrame): the weights of the assets in the minimum volatility portfolio
    """

    min_volatility_portfolio = minVolatility(mean_returns, cov_matrix, constraint_set) #INCLUDE CONSTRAINTS
    min_volatility_returns, min_volatility_std = calculateTotalPortfolioPerformance(min_volatility_portfolio['x'], mean_returns, cov_matrix)
    min_volatility_weights = pd.DataFrame(min_volatility_portfolio['x'], index=mean_returns.index, columns=['weight'])
    min_volatility_weights.weight = [round(i*100,2)for i in min_volatility_weights.weight]
    return min_volatility_returns, min_volatility_std, min_volatility_weights

def getEfficientFrontier(mean_returns, cov_matrix, constraint_set):
    """
    Returns the efficient frontier given the mean returns and covariance matrix
    
    Args:
    mean_returns (pandas.Series): the mean returns of each asset in the portfolio
    cov_matrix (pandas.DataFrame): the covariance matrix of the assets in the portfolio
    constraint_set (list): the constraint set for the asset weights

    Returns:
    max_sharpe_ratio_returns (float): the returns of the maximum sharpe ratio portfolio
    max_sharpe_ratio_std (float): the standard deviation of the maximum sharpe ratio portfolio
    max_sharpe_ratio_weights (pandas.DataFrame): the weights of the assets in the maximum sharpe ratio portfolio
    min_volatility_returns (float): the returns of the minimum volatility portfolio
    min_volatility_std (float): the standard deviation of the minimum volatility portfolio
    min_volatility_weights (pandas.DataFrame): the weights of the assets in the minimum volatility portfolio
    target_returns (list): the target returns for which the optimal asset weights are calculated
    efficient_frontier_list (list): the list of efficient frontier portfolios
    efficient_frontier_weight_list (list): the list of efficient frontier portfolio weights
    """

    # Calculate the maximum sharpe ratio portfolio
    max_sharpe_ratio_returns, max_sharpe_ratio_std, max_sharpe_ratio_weights = getMaxSharpeRatioPortfolio(mean_returns, cov_matrix, constraint_set)

    # Calculate the minimum volatility portfolio
    min_volatility_returns, min_volatility_std, min_volatility_weights = getMinVolatilityPortfolio(mean_returns, cov_matrix, constraint_set)   

    # Calculate an interval of target returns between the minimum volatility portfolio and the maximum sharpe ratio portfolio returns
    target_returns = np.linspace(min_volatility_returns, max_sharpe_ratio_returns, 100)

    #Set up arrays to hold the results for each target return
    efficient_frontier_list = []
    efficient_frontier_weight_list = []

    # Calculate the optimal portfolio for each target return and store the results in the arrays
    for target in target_returns:
        result = efficient_frontier(mean_returns, cov_matrix, target, constraint_set)
        efficient_frontier_list.append((result['fun']))
        efficient_frontier_weight_list.append(result['x'])
    
    return max_sharpe_ratio_returns, max_sharpe_ratio_std, max_sharpe_ratio_weights, min_volatility_returns, min_volatility_std, min_volatility_weights, target_returns, efficient_frontier_list, efficient_frontier_weight_list


# Setup for flask API endpoint
app = Flask(__name__)

# Enable CORS
CORS(app)   


@app.route('/portfolio') # Define the API endpoint
def returnEfficientFrontier():
    """
    Returns the efficient frontier and the optimal portfolios for the given assets and time period
    Request Parameters:
    assets (string): comma separated list of assets
    days (int): number of days to include in the analysis
    includeAllAssets (boolean): whether to include all assets in the portfolio or not

    Returns:
    response (json): the efficient frontier and optimal portfolios
    """

    global timeframe

    # Define the asset weight constraints
    constraints = (0,1)
    if (request.args.get('includeAllAssets') == 'true'):
        constraints = (0.02,1)

    # Extract request parameters
    assets = request.args.get('assets')
    assetNames = assets.split(',')
    days = int(request.args.get('days'))

    # Adjust timeframe based on days passed in request
    timeframe = days
    starttime = endtime - dt.timedelta(days=days)

    # Calculated mean returns and covariance matrix for the assets
    mean_returns, cov_matrix = getAssetData(assetNames, starttime, endtime)

    # Calculate the efficient frontier and optimal portfolios
    max_sharpe_ratio_returns, max_sharpe_ratio_std, max_sharpe_ratio_weights, min_volatility_returns, min_volatility_std, min_volatility_weights, target_returns, efficient_frontier, efficient_frontier_weight_list = getEfficientFrontier(mean_returns, cov_matrix, constraints)

    # Create the data frame of the key portfolios (maximum sharpe ratio and minimum volatility) and populate it with the respective data
    columns = ['Portfolio', 'Returns (%)', 'Volatility (%)',] + [asset for asset in assetNames]
    rows = []
    portfolio_data_frame = pd.DataFrame(columns=columns)
    max_sharpe_ratio_portfolio_data = ['Maximum Sharpe Ratio', round(max_sharpe_ratio_returns*100,2), round(max_sharpe_ratio_std*100,2)] + max_sharpe_ratio_weights['weight'].tolist()
    min_volatility_portfolio_data = ['Minimum Volatility', round(min_volatility_returns*100,2), round(min_volatility_std*100,2)] + min_volatility_weights['weight'].tolist()
    rows.append(max_sharpe_ratio_portfolio_data)
    rows.append(min_volatility_portfolio_data)

    # Create the data frame of all portfolios and populate it with the respective data
    allPortfolios = pd.DataFrame(np.round(efficient_frontier_weight_list,2), columns=assetNames)
    allPortfolios['Returns'] = np.round(target_returns*100,2)
    allPortfolios['Volatility'] = [round(val * 100, 2) for val in efficient_frontier]
    allPortfolios['Sharpe Ratio'] = ((allPortfolios['Returns'] - RISK_FREE_RATE) / allPortfolios['Volatility']).round(2)
    allPortfolios = allPortfolios[['Sharpe Ratio', 'Returns', 'Volatility'] + assetNames]

    # Assign the data rows to the DataFrame
    portfolio_data_frame = pd.DataFrame(rows, columns=columns)

    # Create scatter plots for Maximum Sharpe Ratio and Minimum Volatility points using Plotly
    max_sharpe = go.Scatter(x=[max_sharpe_ratio_std], y=[max_sharpe_ratio_returns], mode='markers', name='Maximum Sharpe Ratio', marker=dict(color='red', size=10))
    min_volatility = go.Scatter(x=[min_volatility_std], y=[min_volatility_returns], mode='markers', name='Minimum Volatility', marker=dict(color='blue', size=10))

    # Create dashed line for efficient frontier using Plotly
    efficient_frontier_line = go.Scatter(x=efficient_frontier, y=target_returns, mode='lines', name='Efficient Frontier', line=dict(dash='dash', color='black'))

    # Create the efficient frontier plot and minimum volatility and maximum sharpe ratio points
    data = [max_sharpe, min_volatility, efficient_frontier_line]
    layout = go.Layout(
        title='Efficient Frontier',
        xaxis=dict(title='Volatility'),
        yaxis=dict(title='Expected Returns'),
        hovermode='closest'
    )
    fig = go.Figure(data=data, layout=layout)

    # Create the response object with the efficient frontier plot and the data frames of the key portfolios and all portfolios
    response = {
        'efficientFrontier': plotly.io.to_json(fig, pretty=True),
        'portfolio': portfolio_data_frame.to_json(orient='records'),
        'allPortfolios': allPortfolios.to_json(orient='records')
    }

    return response

# run the flask app
if __name__ == '__main__':
    app.run(debug=True)