# A data-driven approach to healthcare treatment with smartwatches

The new frontier of preventive medicine is to use specific biomarkers to forecast and prevent disease. By way of blood sugar
forecasts, the project aims to enable diabetics, pre-diabetics, and even non-diabetics to better monitor their blood sugar levels
in real time without the use of expensive continuous glucose monitors or uncomfortable blood glucose meters.

## Project Phases

- Literature Research: Research into different watches, heart rate monitors, smart devices that track body metrics, and/or glucose levels. 
Have a detailed analysis of what are the terms being measured. Consult with experienced data practioners for best practices on code structure.

- Experimentation: Secure funding for the purchase of a device to collect measurements, i.e. Abbott Libre Sense, or Apple Watch/Samsung Gear, etc, and if it is worthwhile self collecting data.
Begin n-of-1 trials while monitoring potential confounding factors.

- Data Preprocessing, EDA and descriptive analysis: With collected data from self trials, begin data modelling approach and decide on fixed variables for which ML/AI can be performed to provide insights
Clean up dataset, input missing values, and begin exploratory EDA on data to decide on best approach.

- Data modelling using classical and deep learning approaches: With the statistical structure of the data established, utilized established models, e.g. ARIMA, EMA, VAR, to forecast changing glucose values.
The use of non-linear methods includingsupervised deep learning using LSTMs, RNNs and GRUs can be considered as well.


## Data

3 trials were conducted, with only analysis conducted on trial 2 and 3.

- Trial 1: Dec 2021: The use of a mobile device for HRV recordings meant large gaps in the time series data, rendering the majority of recordings
unusable. It is stil included for reference.

- Trial 2: Mar 2022: Only 5 days of sampling, results were inconclusive. No walk-forward validation was used. Only classical methods, e.g. ARIMA,
ARIMAX and VARMA were attempted in trial 2.

- Trial 3: April 2022: 8 days of sampling, all data usable for analysis. Walk forward validation used for the majority of models. Supervised deep
learning using LSTMs was attempted on exogenous data to predict endogenous variables. This is also the trial that was 
used in the final presentation and the report.
## Models

A total of 4 notebooks are available, with several models used, 3 classical, 1 deep learning. All model notebooks are located in ~/data/Phase 3 Trial (Mar 2022).

- Naive Mean - ```naive_model.ipynb```: 
With different windows created as a result of a walk-forward split, the naive mean model uses the mean of the last window
as a forecast for the current window. This model will be used as a benchmark for the other more sophiscated models.

- ARIMA - ```ARIMA-Walkforward.ipynb```: 
ARIMA models are classical lag/error models that use past lags and errors, along with seasonal effects (if required) to forecast
future values. In time series analysis, ARIMA models are frequently used due to their easy intepretability. In this notebook,
a classical ARIMA without walk-forward validation is shown, followed by an ARIMAX model (ARIMA + exogenous variables). Finally,
a VARMA (vector ARIMA) model is shown, which treats the multivariate nature of the dataset as independent vectors.

- LSTM - ```LSTM.ipynb```: 
LSTM models are frequently used in time series and classification problems due to their potential to avoid the vanishing gradient
problem. They are a subclass of Recurrent Neural Networks, along with variants such as GRUs. Here, an attempt to forecast endogenous
variables using exogenous HRV attributes is performed. Finally, an all-encompassing model using both HRV and past glucose values
was created and tested, with results far surpassing classical models.

## Run Locally

Clone the project

```bash
  git clone https://github.com/itsdarrenlee/HRV-glucose-modelling-FYP
```

Go to the project directory

```bash
  cd HRV-glucose-modelling-FYP
```

Create and activate new conda virtual environment

```bash
  conda create -n my-test-env && conda activate my-test-env
```

Start up new jupyter notebook server in cmd/terminal

```bash
  jupyter notebook
```

In Jupyter, select your new conda env 'my-test-env' and navigate to your saved repo.
Above steps can also be performed in Visual Studio Code.

**Important**: Before running any models, you must run `preprocessing.ipynb` to save the preprocessed and cleaned dataframes
to your local drive. Without running `preprocessing.ipynb`, all other model notebooks will not run as they rely on a 
cleaned and processed dataset already present in the local directory.

That's it! Have fun with the various notebooks.

