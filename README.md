# Acceleration-of-Jackknife_plus
This repository contains MATLAB code to reproduce the main simulation study presented in the paper:

**“Fast Jackknife+: Accelerating Prediction Interval Construction via Approximate Leave-One-Out Estimators”**

## Description

The goal of this code is to evaluate the performance and efficiency of two methods for constructing prediction intervals in a Ridge regression setting:

1. **Jackknife+** (JK+) using exact leave-one-out (LOO) ridge estimators.
2. **Fast Jackknife+** using approximate LOO ridge estimators (Woodbury formula-based update).

We assess both statistics(mean, median and standard deviation) of:
- **Coverage accuracy**: proportion of true test responses covered by the prediction intervals.
- **Computation time**: runtime for prediction interval construction of each trail.

The code computes the coverage rates and runtimes across multiple simulated datasets with varying predictor dimension \( p \in \{50, 100, 200\} \), while fixing the number of training and test samples to 100 each. 

## File Contents

- `simulation.m`: Main script for running the full simulation study

## How to Run

1. Open MATLAB(The version of MATLAB I use at here is MATLAB 2024b).
2. Ensure all files are in your working directory.
3. Run the `simulation.m` script. This will:
   - Simulate 50 replications for each \( p \in \{50,100,200\} \),
   - Construct JK+ and Fast JK+ prediction intervals for test set,
   - Compute coverage and timing statistics,
   - Save results to Excel sheets.
