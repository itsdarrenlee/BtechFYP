from sklearn import metrics
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import pacf, acf

def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    
    return {"rmse" : rmse,
            "mae": mae,
            "mse": mse,
            "mape": mape,
            "r2": r2}


def adfuller_test(series, signif=0.05, name='', test = 'ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.\n")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.\n")


def cointegration_test(data, alpha=0.05):
    '''
    Cointegration test:
    
    To find out how many lagging terms are required for a TS to become stationary.
    With two or more TS, they are considered cointegrated if they have a statistically significant relationship.
    This means, there exists a linear combination of them that has an order of integration less than that of the individual series.

    - https://en.wikipedia.org/wiki/Cointegration
    - http://www-stat.wharton.upenn.edu/~steele/Courses/434/434Context/Co-integration/Murray93DrunkAndDog.pdf
    - https://en.m.wikipedia.org/wiki/Johansen_test
    - https://en.wikipedia.org/wiki/Error_correction_model

    null hypothesis: no cointegrating equations, alternate hypothesis: at least 1 cointegrating relationship
    
    '''
    out = coint_johansen(data, -1, 5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(data.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False, maxlag=12):    
    '''
    ### Interpreting time series correlation
    Refer to:
    >https://stats.stackexchange.com/questions/133155/how-to-use-pearson-correlation-correctly-with-time-series

    Granger casuality testing only applies to stationary intervals.
    - Null hypothesis: HRV feature (x) does not explain variation in (y) glucose.
    - Alternative hypothesis: HRV feature (x) has an effect on glucose with a 95% confidence interval that a change in x causes a response in 'glucose' (y)
    '''
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def scale_df(train_df, val_df, test_df, scaler=MinMaxScaler(),
             drop_dependent=True, dep_var='glucose'):
    """Scales using defined scaler and reshape dataframe in preparation for TimeseriesGenerator input

    Args:
        train_df (Pandas DataFrame): Train set, including all variables
        val_df (Pandas DataFrame): Validation set, including all variables
        test_df (Pandas DataFrame): Test set, including all variables
        scaler (Sklearn Scaler Object, optional): Defined Scaler to use. Defaults to MinMaxScaler().
        drop_dependent (bool, optional): Drop dependent, i.e. 'Glucose' if desirable. Defaults to True.
        dep_var (str, optional): Dependent variable. Defaults to 'glucose'.
        
    Returns:
        Numpy Arrays: Tuple of X_train, X_validation, X_test, y_train, y_validation and y_test
    """

    if drop_dependent: # drop dependent variable, 'glucose' if desirable
        train_df = train_df.drop(dep_var, axis=1)
        val_df = val_df.drop(dep_var, axis=1)
        test_df = test_df.drop(dep_var, axis=1)
    
    print(f"Shape of train: {train_df.shape}, shape of validation: {val_df.shape}, \
 shape of test: {test_df.shape}") # sanity check for dataframe shape if var was dropped

    # independent variable scaling
    X_train_scaled = scaler.fit_transform(train_df)
    X_val_scaled = scaler.transform(val_df)
    X_test_scaled = scaler.transform(test_df)

    # dependent variable scaling
    y_train_scaled = scaler.fit_transform(train[dep_var].
                                        values.reshape(-1, 1))
    y_val_scaled = scaler.transform(val[dep_var].
                                        values.reshape(-1, 1))
    y_test_scaled = scaler.transform(test[dep_var].
                                        values.reshape(-1, 1))
    
    # return as raw numpy arrays
    return (X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled,
            y_val_scaled, y_test_scaled, scaler)    


def create_corr_plot(series, lags=30, plot_pacf=False):
    corr_array = pacf(series.dropna(), alpha=0.05, nlags=lags) if plot_pacf\
        else acf(series.dropna(), alpha=0.05, nlags=lags)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    fig = go.Figure()
    [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f') 
     for x in range(len(corr_array[0]))]
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                   marker_size=5)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
            fill='tonexty', line_color='rgba(255,255,255,0)')
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,(0.7*lags)])
    fig.update_yaxes(zerolinecolor='#000000')
    
    title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig.update_layout(title=title)
    fig.show()