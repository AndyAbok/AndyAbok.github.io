---
layout: post
title: The Levenberg-Marquardt algorithm
date: 2025-02-03 00:00:00 +0300
categories: [Capital Markets]
tags: [Capital Markets]
math: true
---
# Notes On What Makes An Asset Useful

## Introduction
This write up is on notes the paper What makes an asset useful by Dr. Yves-Laurent Kom Samoa and  Dr. Dieter Hendricks ,The author introduces new methodologies on how to determine which assets to invest in focusing on four key aspects.

Background: The paper discusses the key qualitative question in the investment process of how to assess the usefulness of a new candidate asset represented as a time series of returns.

**Justification of the problem** : The usefulness of an asset can only be determined relative to a reference universe of assets and/or benchmarks. The paper argues that there are four features that the time series of returns of an asset should exhibit for the asset to be useful to an investment manager. Most models proposed by MPT assume Normality of returns which is not always the case.

**Objectives** : The objective is to provide a precise method for assessing the usefulness of a new candidate asset. The paper proposes four criteria for determining the usefulness of an asset: incremental diversification, returns predictability, tail risk mitigation, and suitability for passive investment. The paper also provides scalable algorithmic tests for each criteria.


## Methods 
The  study design for the paper was an observational study. The authors observe and analyze the relationships between assets and portfolio diversification.

The paper used secondary data that is historical data on various asset classes to analyze the benefits of incremental diversification. They calculate the information-adjusted correlation between assets and use simulation analysis to compare the performance of traditional diversification strategies with and without incremental diversification.

By observing and analyzing historical data, the authors are able to provide empirical evidence to support their arguments and provide insights into the benefits of incremental diversification for portfolio management.


## Quantitative Methods
The authors Classify the four key criteria a new asset should exhibit to be considered useful to an investment manager into two primary and two secondary.


For each of the approaches they explore three main approaches in quantifying most of the measures Incremental measure Namely,


1. Gaussian method(Conditional  Variance Method) 
2. Beyond Gaussian Method(Conditional Entropy) 
3. Differential Entropy method.

The authors framework assumes that an asset is fully characterized by its stream of returns, that two assets that have identical time series of returns are identical for all practical investment purposes, and consequently they solely rely on an asset’s time series of returns to answer the foregoing question.

### 1. Incremental diversification
The foremost reason to select an asset is if it diversifies an investment managers portfolio, Which is not always the case as some assets provide high potential than others. 

Let $𝐴=(𝐴_1, …, 𝐴_𝑝)$ be the new universe of new assets for which no asset is fully determined by the others, Let $𝑦_𝑡={(𝑦_𝑡^1, …,𝑦_𝑡^𝑝 )},𝑝>1$ be the vector valued time series returns of assets in the new universe and $𝑥_𝑡$ the time series of return values of the existing reference pool of assets, then the amount of diversification the new universe of assets add to the existing one is given as 

$$
I𝐷(𝐴;𝑃)=  \frac{1}{(ℎ({𝑦_𝑡 })+ℎ({𝑥_𝑡 })−ℎ({𝑥_𝑡,𝑦_𝑡 }))}
$$

The proposed model by the authors is based on the principle of maximum entropy pioneered by E.T. Jaynes in [22,23] it stipulates that, when faced with an estimation problem, among all models that are consistent with empirical evidence, one should always choose the one that is the most uncertain about everything other than what has been observed.

Theorem: 

Let $𝑧_𝑡$ be a stationary $\mathbb{R}^n$ valued discrete time stochastic time process. Among all stationary process whose(matrix valued) autocovariance functions coincide with that of $𝑧_𝑡$ from lag ℎ=0 to lag ℎ=𝑝 
$$
ℎ(𝑧_𝑡)= \frac{𝑛}{2}  log_2⁡2𝜋𝑒+\frac{1}{2} log_2
\begin{bmatrix}\frac{det(\sum_{p})}{det(\sum_{p-1})}]
\end{bmatrix}
$$

Where:

$\sum_{p}$ is the block matrix such that 
$\sum_{p} [𝑖,𝑗]=cov(𝑧_{t+1},𝑧_{𝑡+𝑗})=𝐶(𝑖−𝑗)$
In cases of two assets the paper proposes information adjusted correlation coefficient:

$$
ID({y_t};{x_t}) = \frac{-2}{log(1 - corr(y_t,x_t)^2)}
$$

$$
corr_A(A,B) = corr(A,B)\sqrt{1 - 2\frac{-2}{ID(A;B)}}
$$

#### Illustration
Estimation of the entropy rate of an AR(1) process with Student-t noise with ν degrees of freedom, scale parameter chosen so that the innovation 
process is unit standard deviation,and for a sample size T.
The relative error is defined as  $100 ×  \frac{(ℎ − \hat{ℎ})}{|h|}$ 

### 2.Sufficiently predictable returns 

A time series of returns $y_t$ can be considered sufficiently predictable if there exists a stream of information available at time t that reduces the uncertainty about future values at time  $t+p$, $p>0$ 

It would be impractical to determine whether future values of a return series can be predicted using any information that currently exists whether we have access to it or not Instead, the authors focus on quantifying whether a time series can be predicted using data that we do have access to, starting with all past values of the time series, and then generalizing to any stream of information that can be accessed.


When $y_t$ is strongly stationary, the entropy rate $ℎ(y_t)$ always exists and it can be shown that $h(y_t|y_{t-1},...,y_1)$ decreases with t thus:

$$
PR(\{y_t\}) = h(y_t) - h(\{y_t\})
$$

Can be regarded as the maximum reduction in the uncertainty of the return at any point in time t that one can achieve by knowing all returns prior  to t which makes it suitable for quantifying predictability of returns.PR is the measure of auto predictability of time series 


### 3. Mitigate Tail risk.
Tail risk is the risk of extreme losses beyond a certain threshold, the paper argues that a new asset that can mitigate tail risk can be useful to an investment manager as it helps to reduce the overall risk of the portfolio by reducing the idiosyncratic risk, that is the risk caused by the market i.e extreme losses. A new asset that has large idiosyncratic moves affects the tail behavior of the portfolio so lighter tailed assets would be preferred.

$$
kurt(y_t) = \mathbb{E}\begin{bmatrix}(\frac{y_t - E(y_t)}{\sqrt{Var(y_t)}})^4
\end{bmatrix}
$$

The Authors extended this to define kurtosis as $kurt(x_t) = \mathbb{E}(\phi(x_t)^2)$ where 
$$
\phi(x_t)= (x_t - E(x_t))^T Cov(x_t,x_t)^{-1}(x_t - E(x_t))
$$

The tail ratio would now be given by $TR(x_t) = \frac{\mathbb{E}(\phi(x_t)^2)}{n(n+2)}$  where n is the degree of freedom. 
To quantify the impact of a new asset on the tails of a reference pool is they propose comparing the tail ratio of the reference pool with and without the new asset, for instance through the difference.

### 4. Sustainability for passive investment.
An asset should appreciate overtime to be able to earn income to the investor. A passive investment can be useful to an investment manager as it allows for a more hands-off approach to portfolio management.
Whether a new asset is suitable for passive investment or not boils down to whether one can achieve a decent level of risk-adjusted returns without changing the investment decision too often.


The authors extend the sharp ratio to come with the bidirectional sharp ratio of an asset. Which they define as:

$$
BSR(A) = \frac{\mathbb{E}|(y_t)| - r_c - r_f}{\sqrt{Var(y_t)}}
$$

Where:

 $\mathbb{E}(y_t)$ is the Expected gross return of the asset 
 
 $r_c$  is the total operating cost incurred per time period 
 
 $r_f$ is the risk free rate.
 
 
The bidirectional Sharpe ratio therefore represents a measure of the best expected net excess return per unit of risk that can be obtained by passively investing in an asset. Therefore  quantifying suitability for passive investment requires accurately estimating a bidirectional Sharpe ratio, and quantifying its similarity to those of assets that appeal to large asset managers.


### Conclusions
The paper identified four key criteria a new asset should exhibit to be considered useful to an investment manager, two primary and two secondary and all four are independent from the investment manager’s asset allocation strategy.


As primary criteria, the authors propose that, to be useful, a new asset should sufficiently diversify the pool of assets and factors the investment manager already has access to, and the new asset’s returns time series should be sufficiently predictable.


The authors also propose as a secondary criteria that, to be useful, a new asset should not have an excessive adverse impact on the tails of assets the investment manager currently trades, and it should be suitable for passive investment.


The authors propose a mathematical framework for quantifying all four criteria, and provide scalable algorithmic solutions.


### Empirical Findings 

1. The study finds evidence that daily returns of S&P 100 constituents exhibit a non-linear and/or cross-sectional and temporal relationship that pairwise correlation cannot capture, but mutual information timescale can.
2. Cross-asset class diversification is more effective than within-asset class diversification, but the degree of incremental diversification varies depending on the asset classes involved. For instance, using currencies to diversify U.S. futures adds more diversification than trading more U.S. futures.
3. Time series of asset returns have memory and are predictable, with currencies being less predictable than stocks and futures overall, and the predictabilities of futures varying significantly depending on the underlying.
4. Currencies have the heaviest tails in isolation, but adding them to a basket of U.S. blue chip stocks often has a positive impact on tails, while adding blue chips to currencies on average has a negative effect on tails.
Foreign currencies are not suitable for passive investment, suggesting that active management is necessary to make money trading currencies.


## References
* Miguel de Cervantes Saavedra. El Ingenioso Hidalgo Don Quijote de la Mancha, volume 1. 1605. 
* Harry Markowitz. Portfolio Selection. The Journal of Finance, 7(1):77–91, 1952.
* Thomas M Cover and Joy A Thomas. Elements of Information Theory. John Wiley & Sons, 2012. 
* Shunsuke Ihara. Information Theory for C
* Edwin T Jaynes. Information Theory and Statistical Mechanics.i. Physical review, 106(4):620, 1957. 
* Edwin T Jaynes. Information Theory and Statistical Mechanics. ii. Physical review, 108(2):171, 1957
