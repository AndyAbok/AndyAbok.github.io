---
layout: post
title: Security Price Forecasting using Hidden Markov Model 
date: 2024-10-08 00:00:00 +0300
categories: [Predictive Modelling]
tags: [python,capital markets]
math: true
canonical_url: https://www.innova.co.ke/hidden-markov-model/
---

### *A case study of predicting the prices of Equity Bank Kenya*

This post was originally published on the <a href="https://www.innova.co.ke/hidden-markov-model/" target="_blank">Innova official blog</a>.

## 1.) Introduction
Prediction or forecast is a statement about a future event or data.They are often, but not always, based upon experience or knowledge.This is the definition of prediction in wikipedia

In our every day life we try to some extent to anticipate the future and this enables one to know what to expect so as to organize themselves to succeed in whatever it is they are doing.

All fields are racing to try and predict the future and models have been written and resources pumped into predicting the future. The same is true for investments prices are being predicted to enable for securities selection to invest in,For portfolio optimization to be made possible the expected returns should also be predicted.The main reasons for forecasting in the financial markets are to maximize returns for the investor and minimize risk as well.

Of course ,all this goes without saying that for a security's prices to be predicted with some level of certainty the asset should not be extremely noisy, nonetheless the financial market is partly deterministic but chaotic, and partly a stochastic process and so in this case the forecast values should not be counted on with much confidence.This means that the financial market includes randomness which make it hard to make accurate predictions.

In this article we will try to forecast prices by applying Hidden Markov Models into financial markets.


## 2.) Hidden Markov Model

HMM is an instance of the state space model in which the latent variable are discrete.What's this state space model? State-space models are models that use state variables to describe a system by a set of first-order difference equations.This is similar to the principles of  Markov chains however in this case the latent states are assumed to be the dictator of the observed values.

At each time step, the latent state governs which observation is emitted, making the sequence of observed states a probabilistic function of the hidden states.

### Mixture Models and Latent Variables

if you examine a single time slice of the model, we see that it corresponds to a mixture distribution, with component densities given by 
$p(x|z)$.As in the case of a standard mixture model, the latent variables are the discrete multinomial variables $𝑧_𝑛$ describing which component of the mixture is responsible for generating the corresponding observation $𝑥_𝑛$. 

The Specification of this probabilistic model is completed by defining the conditional distributions of the observed variables 
$𝑝(𝑥_𝑛 |𝑧_𝑛,𝜑)$ , where $𝜑$ is a set of parameters governing the distribution. These are known as emission probabilities, They are mostly given by Gaussians if the elements of $x$ are continuous variables, or by conditional probability tables if $x$ is discrete.  

![](/assets/images/MarkovModel.png)

### Defining the Model

We can define a particular sequence of visible/observable states as
$$
V^T = {v(1),v(2),v(3)...v(T)}
$$
These are the states that we can directly observe, while the hidden states remain unobservable.The model can then be defined by a set of parameters $\theta$, Which usually includes:

* Transition probabilities
* Emission probabilities
* Initial State probabilities

so in any state $s_t$ we have a probability of emitting a particular visible state $v_k(t)$.Having access to only visible states while $s_𝑡$‘s are unobservable, such is a Hidden Markov Model and the Network is refereed to as Finite-State Machine. 

### Building Algorithms for Hidden Markov Models

There are three main algorithms used for inference in HMMs:

1. The Virterbi algorithm - Finds the most likely sequence for the hidden process.

2. Forward-backward Algorithm - calculates the probability distribution for the hidden states at every location.
3. Baum-Welch algorithm - Estimates any parameters that occur in the model using observed data.

Let $𝑦_𝑡$  represent the observed value of the process at time $𝑡$  for $𝑡=1,…𝑇$, $\theta_t$ represent the value of the hidden process at time $t$.Let $\phi$ represent the parameters necessary to determine the probability distribution for $𝑦_𝑡$ given $\theta_t$ and $\theta_{t-1}$.In our example, the observed values $𝑦_𝑡$ will be either an increase or decrease in the security prices and the hidden states $\theta_t$ will determine the probability distribution of observing the different values. 


The model can be described  by sets of probability distributions 

* $𝑝(𝑦_𝑡│\theta_t,\phi)$ - representing the likelihood of observing $y_t$ given the hidden state.

* $𝑝(\theta_t│\theta_{𝑡−1},\phi)$ - Representing the transition between hidden states over time.

> Key to Note is $𝑦_𝑡$ are conditionally independent given the set of hidden states $\theta_t$ and each hidden state only depends on its neighbors $\theta_{t-1}$.   


### Maximum Likelihood in HMMs

The goal of HMM inference is to find the path that has the maximum probability in the HMM given the observed sequence X.We can define the model Mathematically 

$$
𝐻𝑀𝑀= (𝑆,𝑂,𝐴,𝐵,\pi) 
$$

Where:

* $S$ : Set of hidden states
* $O$ : Set of observations
* $A$ : Transition probabilities between states
* $B$ : Emission probabilities (likelihood of observations given hidden states)
* $\pi$ : Initial state distribution


Given a sequence of observations, the parameters of an HMM can be determined using ***maximum likelihood estimation***. The likelihood function is obtained from the joint distribution of the observations and hidden states by marginalizing over the latent variables. 

Expectation maximization algorithm is used for maximizing the likelihood function in HMM.The algorith iterates between two steps:

1. **E-Step (Expectation)**: Compute the expected value of the hidden states given the current model parameters.
2. **M-Step (Maximization)**: Update the model parameters to maximize the likelihood based on the expectations computed in the E-step.

## Methodology 

The methodology used here is heavily borrowed from <a href="https://d1wqtxts1xzle7.cloudfront.net/52961747/Stock_market_prediction_using_hidden_markov_models-libre.pdf?1493840494=&response-content-disposition=inline%3B+filename%3DStock_market_prediction_using_hidden_mar.pdf&Expires=1728298235&Signature=N-t9Z-NMRIXBP3BTRTqzvL~pB9Iz-dwKL86f~PfE-7dRxICFhi~xYN8jonhbZZ93aJnhbdHBft1DXKcYbn~~1qks~fxgMaiCWSALTbHw--JIsuUEeKqxIgvan0z0psQy82aL-ww2PsygiQCi36u324UGtcFgG8OBdOQEE92hppbaQjweZayZxFR6YfqttrzKRoQ9wQcQaYpW6~R5zSVne7qUngyf5CvQRsl6JfIBU5GNnwZlEq8~Sn1hy4nP~zIKYKaF3Lh8xjbpQZ4S5ezSOFf0ChdScHpnubOgVYDOlcaMUsGPL-z1kTOfqvgjIgCYFjm0oFBHaXfErf87f3L4Ow__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA" target="_blank">(Gupta,2012)</a> in his paper "Stock Market Prediction Using Hidden Markov Models," with some modifications to improve accuracy and flexibility.

1. **Data Processing**: The market data is has 4 attributes: opening price, closing price, high price, and low price. 
    ```
     MarketDate                      Name  Close    Low   High  Open
    0 2018-01-02  Equity Bank Ltd Ord 0.50  40.00  39.50  40.25  37.7
    1 2018-01-03  Equity Bank Ltd Ord 0.50  40.00  40.00  40.50  40.0
    2 2018-01-04  Equity Bank Ltd Ord 0.50  40.00  39.75  40.50  40.0
    3 2018-01-05  Equity Bank Ltd Ord 0.50  40.25  39.75  40.50  40.0
    ```

The model focuses on fractional changes in these values as its observations to capture the security's behavior over time.

```python
def getFeaturesVector(data:pd.DataFrame):
    closePrice = np.array(data['Close'])
    openPrice = np.array(data['Open'])
    highPrice = np.array(data['High'])
    lowPrice = np.array(data['Low'])

    fracChange = (closePrice - openPrice) / openPrice
    fracHigh = (highPrice - openPrice) / openPrice
    fracLow = (openPrice - lowPrice) / openPrice
    return np.column_stack((fracChange, fracHigh, fracLow))

fractionalChanges = getFeaturesVector(securityPrices)
```

2. **Training and Modeling with HMMs**: A continuous Hidden Markov Model (CHMM) is employed to model the stock data as a time series. The HMM is characterized by Hidden states Emission probabilities that are elaborated in the introduction. 

The HMM is trained using historical market data. The model parameters are estimated  using the Baum-Welch algorithm.

The HMM model is assumed to have 6 states namely:
- very low
- low
- moderately low
- moderately high
- high
- very high

    ```python
    nStates = 6
    nComponents = 5 
    
    # Scaling the data for training
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(fractionalChanges)
    
    hmm = GaussianHMM(n_components=nStates, covariance_type="diag", n_iter=100)
    hmm.fit(scaledData)
    ```

4. **Forecasting**: 
The key difference from Gupta's (2012) methodology lies in how the future prices are predicted. In this implementation, we perform MAP estimation over a grid of possible fractional changes, high prices, and low prices to maximize the likelihood of each predicted outcome Unlike the original methodology that simply predicts based on the most likely state transition, the approach used here generates a set of possible outcomes (using ranges for fractional changes) and scores each outcome based on its likelihood, selecting the one with the highest score.This allows the model to account for uncertainty and provides more accurate multi-day predictions.

    ```python
    def map_predict(hmmModel, previousData, last_open_price, nDays, 
                nStepsFracChange=30, nStepsFracHigh=10, nStepsFracLow=10):
    
        # Define ranges for fractional changes
        fracChangeRange = np.linspace(-0.1, 0.1, nStepsFracChange)
        fracHighRange = np.linspace(0, 0.1, nStepsFracHigh)
        fracLowRange = np.linspace(0, 0.1, nStepsFracLow)
        
        # Generate all possible outcomes as combinations of fracChange, fracHigh, fracLow
        possibleOutcomes = np.array(list(itertools.product(fracChangeRange, fracHighRange, fracLowRange)))
        
        predicted_close_prices = []
        openPrice = last_open_price  
    
        for day in range(nDays):
            outcomeScores = []
            
            # Loop through all possible outcomes and compute their likelihood scores
            for possibleOutcome in possibleOutcomes:            
                total_data = np.row_stack((previousData, possibleOutcome))
                outcome_score = hmmModel.score(total_data)
                outcomeScores.append(outcome_score)
            
            # Select the outcome with the maximum likelihood score
            bestOutcome = possibleOutcomes[np.argmax(outcomeScores)]
            
            # Calculate predicted closing price using the best outcome
            predicted_close = openPrice * (1 + bestOutcome[0])  
            predicted_close_prices.append(predicted_close)
            
            openPrice = predicted_close
            previousData = np.vstack([previousData, bestOutcome])

    return np.array(predicted_close_prices)
    ```

To further enhance accuracy, we simulate multiple price paths and compute the average forecasted price across simulations.


```python
def simulatePricePaths(hmm, scaledData, last_open_price, n_future_days, latencyDays=10, nSimulations=100):
    simulations = []
    for _ in range(nSimulations):
        estimatedPrices = map_predict(hmm, scaledData[-latencyDays:], last_open_price, n_future_days)
        simulations.append(estimatedPrices)
    
    avgPredictedPrices = np.mean(simulations, axis=0)
    return avgPredictedPrices

predicted_close_prices = simulatePricePaths(hmm, scaledData, last_open_price, n_future_days, nSimulations=10)
```
### Model Evaluation.
The model is Evaluated  using Mean Absolute Percentage Error (MAPE).The model gives a MAPE of 36.77%,clearly there is considerable room for improvement to bring the model to a MAPE of below 10 or 20% to be able to rely on the forecast. 

Lets take a look at the  ahead and visualize the predicted prices against the actual prices.
![](/assets/images/hmmForecast.png)

## Conclusions

we used a Hidden Markov Model (HMM) to predict the  prices of Equity Bank Kenya. The model was developed based on the methodology from Gupta (2012), with some modification to improve prediction accuracy.we used a grid search combined with Maximum a Posteriori (MAP) estimation to maximize the likelihood of predicted outcomes.

The model achieved a **Mean Absolute Percentage Error (MAPE) of 36.77%**, this indicates moderate predictive accuracy. While the model was able to capture general price trends,as n future days went further the predictions became poor diverging from the actual prices as shown in the graph above.This does clearly suggest   that the predictions are not highly reliable particularly in the volatile and stochastic nature of financial markets.

While the current model shows potential, there is considerable room for improvement in order to reduce prediction error and make the forecasts more reliable for decision-making in financial markets. 


### References.
* Gupta, A., & Dhingra, B. (2012, March). Stock market prediction using hidden markov models. In 2012 Students Conference on Engineering and Systems (pp. 1-4). IEEE.

* Bishop, C. M., & Nasrabadi, N. M. (2006). Pattern recognition and machine learning (Vol. 4, No. 4, p. 738). New York: springer.




