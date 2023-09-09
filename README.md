# Portfolio Optimization
Finding optimal portfolio asset allocation based of different criteria

### 1.  Concepts

#### a.  Modern Portfolio Theory (MPT)

-   Risk vs Return

    -   Risk measure = standard variation of return

    -   For a desired level of return, there exist a portfolio that offer the lowest level of risk

-   Efficient Frontier & Optimal Asset Allocation

    -   Efficient Frontier = graphical representation of all possible combinations of risky assets that maximize (minimize) expected return (risk) for a given level of risk (expected return)

    -   Portfolios on the efficient frontier are considered optimal (no other portfolios exist with a higher expected return for the same level of risk)

    -   **Optimal portfolio asset allocation = minimize negative Sharpe ratio (maximize returns for a given level of risk)**

#### b.  Global Minimum Variance Portfolio (GMVP)

-   Key takeaways

    -   Aim: Construct a portfolio from specified asset that offer **minimum variance** (risk)

    -   Should be used when a portfolio of "high return asset" are selected

-   GMVP vs MPT

    -   **Expected return is not considered**

    -   Often consist of asset with negative correlations, combined with derivatives for hedging

#### c.  Risk Parity Portfolio

-   Key takeaways

    -   Aim: Construct a portfolio with each asset contribute the same amount of risk to the portfolio

    -   Want to achieve "true diversification", not in erms of asset classes, but in terms of risk

-   Risk Parity vs MPT

    -   Leverage is often used on low-volatility asset

    -   The allocation is often dynamic, as risk (volatility) changed over time

### 2.  Python Implementation

#### a.  Coding & Algorithms

-   Aim: use optimization algorithm to find the "optimal" allocation of a portfolio with a specified assets

    -   **Sequential Least Squares Quadratic Programming** (SLSQP) algorithm will be used

    -   Constraint: The sum of all asset weights in the portfolio should be equal to 1.

    -   Boundary: Each individual asset weight should be between 0% and 100%.

-   Optimization Metric

    -   MPT: **Sharpe Ratio**

    -   GMVP: **Portfolio Variance**

    -   Risk Parity: **Squared Deviation of Risk Contributions from equal contribution**

-   Input

    -   Tickers: [XLK, XLC, XLY, XLE, XLI, XLB, XLF, XLRE, XLP, XLU, XLV, FXI, GDX]

    -   Time Period: Use 10 years of historical data 2013/08/10 -- 2023/08/10)

    -   Risk Free Rate: 2%

#### b.  Setups

-   Programming details

    -   Required Python packages: yfinance, numpy, pandas, matplotlib, scipy

    -   Program outputs: Portfolio Allocation + Graph plotting the performance of portfolios

    -   A "hard cap" of 0.2 (20%) is put in for the maximum allocation of individual single ticker

    -   Regularization is used for MPT portfolio after tuning

-   Execution Guidelines (for Linux & VS code)

    -   S1: Go to your project folder that you store the script (assuming located at **/folder\_locations**) using **cd /folder\_locations**

    -   S2: Create & activate virtual environment

        -   **\> virtualenv -p python3.9 var\_analysis**

        -   **\> source /folder\_locations/var\_analysis/bin/activate**

    -   S3: Install libraries & provide permissions

        -   **\> pip install yfinance pandas matplotlib numpy scipy**

        -   **\> sudo chmod -R 777 Var\_Historical\_Method.py**

    -   S4: Run the script with the "Run Python file" button in VS Code

### 3.  Results

#### a.  Portfolio Allocation & Performance

 
![Use this template](https://github.com/erikpham95/Portfolio_Optimization/blob/main/Pic/Pic1.png)

![Use this template](https://github.com/erikpham95/Portfolio_Optimization/blob/main/Pic/Pic2.png)


#### b.  Comments & Observations

-   Multiple iterations is needed for SLSQP to find the "best" local
    optima required.

-   Sharpe Ratio of GMVP calculated is higher than of MPT. More
    investigation may be needed.

-   Without regularization & "hard cap", MPT approach resulted in
    single-stock portfolio. More investigation may be needed.
