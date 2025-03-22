# Multi-Kernel Regression (MKR) Trading System

A sophisticated algorithmic trading system that uses various kernel regression methods to identify market trends and generate trading signals.

## Overview

This trading system applies multi-kernel regression techniques to financial market data, particularly focusing on leveraging different kernel functions to identify market trends and generate trading signals. The system is split into two separate implementations:

1. **Long-only version** (this repository): Focuses exclusively on long positions, optimized for uptrend capture
2. **Short-only version** (separate repository): Specialized for short positions, designed for downtrend environments

Both implementations share the same core multi-kernel regression algorithm but are optimized separately for their respective trading directions.

## Features

- **Multiple Kernel Functions**: Implements 8+ kernel types including Gaussian, Logistic, Cosine, Laplace, Silverman, Cauchy, and LogLogistic
- **ATR-Based Position Sizing**: Uses Average True Range for dynamic position sizing and risk management
- **Flexible Exit Strategies**:
  - Take profit targets based on ATR multiples
  - Stop losses with configurable ATR multiples
  - Time-based exits with configurable timeout periods
  - Opposite signal exits
  - End-of-day exits
  - Moving average crossover exits
  - Stochastic exits
- **Comprehensive Backtesting**: Parallel processing capabilities to test thousands of parameter combinations
- **Slippage Modeling**: Incorporates realistic slippage assumptions to model execution costs

## Parameter Optimization

The system can optimize across multiple parameters:
- Kernel type
- Bandwidth
- Difference threshold
- Consecutive signal requirement
- ATR length
- Take profit levels (ATR multiples)
- Stop loss levels (ATR multiples)
- Trade timeout periods
- Moving average periods
- Stochastic indicator parameters

## Results

The optimization process identifies the most profitable parameter combinations across different market conditions and timeframes. The system shows strong performance across various market conditions, as visualized in the performance charts:

![Performance Chart](images/performance.html)
![Long vs Short Performance](images/long_vs_short_SMAs.html)

The repository includes visualizations that compare:
- Long vs. short strategy performance
- Different parameter combinations
- Performance across various market regimes
- Cumulative return profiles

## Usage

1. Configure your desired parameter ranges in the script
2. Set the `USE_ITERTOOLS` flag to either:
   - `True`: To explore the full parameter space
   - `False`: To test only the top-performing parameter combinations from a previous run
3. Adjust the paths for data input and result output
4. Run the script to perform the backtesting

## Requirements

- Python 3.x
- pandas
- numpy
- talib
- pandas_ta
- polygon-api-client (for live data)
- itertools and concurrent.futures (for parallel processing)

## Future Development

- Real-time trading implementation
- Additional exit strategy optimizations
- Machine learning integration for adaptive parameter selection
- Cross-asset class testing

## License

[MIT](LICENSE)
