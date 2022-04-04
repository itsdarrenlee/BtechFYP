from sklearn import metrics
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

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

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series"""
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
             drop_dependent=True, dropvar='glucose',
             dep_var='glucose', return_as_np=True):
    ''' 
    Scales train, val and test dataframes
    
    RETURN: 
    
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, \ 
    y_test_scaled
    
    OR
    
    train_df, val_df, test_df
    
    '''

    if drop_dependent:
        train_df = train_df.drop(dropvar, axis=1)
        val_df = val_df.drop(dropvar, axis=1)
        test_df = test_df.drop(dropvar, axis=1)

    # independent variables
    X_train_scaled = scaler.fit_transform(train_df)
    X_val_scaled = scaler.transform(val_df)
    X_test_scaled = scaler.transform(test_df)

    # dependent variables
    y_train_scaled = scaler.fit_transform(train_df[dep_var].
                                        values.reshape(-1, 1))
    y_val_scaled = scaler.transform(val_df[dep_var].
                                        values.reshape(-1, 1))
    y_test_scaled = scaler.transform(test_df[dep_var].
                                        values.reshape(-1, 1))
    
    # return as pd dataframe for debugging or raw numpy arrays
    if return_as_np:
        return (X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled,
                y_val_scaled, y_test_scaled)
    else:
        if drop_dependent:
            # add glucose back to df
            train_df_scaled = pd.DataFrame(index=train_df.index,
                                           columns=train_df.columns,
                                           data=np.hstack((X_train_scaled,
                                                           y_train_scaled)))
            val_df_scaled = pd.DataFrame(index=val_df.index,
                                          columns=val_df.columns,
                                        data=np.hstack((X_val_scaled,
                                                        y_val_scaled)))
            
            test_df_scaled = pd.DataFrame(index=test_df.index,
                                          columns=test_df.columns,
                                        data=np.hstack((X_test_scaled,
                                                        y_test_scaled)))
        else:
            train_df_scaled = pd.DataFrame(index=train_df.index, columns=train_df.columns,
                                        data=X_train_scaled)
            
            val_df_scaled = pd.DataFrame(index=val_df.index, columns=val_df.columns,
                                        data=X_val_scaled)
            
            test_df_scaled = pd.DataFrame(index=test_df.index, columns=test_df.columns,
                                        data=X_test_scaled)
        
        # return both df
        return train_df_scaled, val_df_scaled, test_df_scaled
    
    
class WindowGenerator():
    '''
    Generate windows from train, validation and test set
    '''
    def __init__(self, input_width, label_width, shift,
                train_df, val_df, test_df,
                label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                                enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
            return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
    
    def plot(self, model=None, plot_col='glucose', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [5Min]')

    