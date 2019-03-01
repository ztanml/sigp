# Finite-Sample Integral Gaussian Processes
A dual construction of Gaussian processes with sample paths in a given reproducing kernel Hilbert space.

## Comparing the SIGP and the standard Gaussian process on Real-Life Data
### Example 1: Classification of the Arcene cancer data
Data source: https://archive.ics.uci.edu/ml/datasets/Arcene

In Matlab:
```
>> demoSIGPArcene
Loading the data ...
Classifying with SIGP ...
F1 score:0.85714
```

For comparison, the standard GP based on GPML Toolbox (http://www.gaussianprocess.org/gpml/code/matlab/doc/) yields a lower F1 score 0.82353. To verify, add GPML to the Matlab PATH, and run demoGPArcene.m. 

### Example 2: Prediction on Boston housing data 
Data source: https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

In Matlab:
```
>> demoSIGPHousing
Loading housing data ...
Training SIGP ...
Mean squared error:28.1999
```

For comparison, the standard GP based on GPML Toolbox (http://www.gaussianprocess.org/gpml/code/matlab/doc/) yields a much larger mean squared error 93.1109. To verify, add GPML to the Matlab PATH, and run demoGPHousing.m. 


## References
More details can be found in our paper: https://arxiv.org/abs/1802.07528
