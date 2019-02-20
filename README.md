# Finite-Sample Integral Gaussian Processes
A dual construction of Gaussian processes with sample paths in a given reproducing kernel Hilbert space.

## Comparing the SIGP and the standard Gaussian process on Real-Life Data
### Example 1: Classification of the Arcene cancer data
Data source: https://archive.ics.uci.edu/ml/datasets/Arcene

```matlab
% Example usage for the classification of Arcene data

disp("Loading the data ...");
feaTrain = load('data/arcene_train.data');
feaTest  = load('data/arcene_valid.data');
gndTrain = load('data/arcene_train.labels');
gndTest  = load('data/arcene_valid.labels');

fea = [feaTrain;feaTest];
fea = fea - mean(fea);
fea = fea./max(std(fea),1e-12);
feaTrain = fea(1:100,:);
feaTest = fea(101:end,:);

disp("Classifying with SIGP ...");
hyp = sigp(feaTrain,gndTrain,1,'covkfn', 'sigp_rbf',...
    'covkpar',362.46,'lambda',1.1863e-05);

disp("F1 score:" + num2str(F1score(sign(hyp.f(feaTest)),gndTest)));
```

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

```matlab
disp('Loading housing data ...')
housedata = load('data/housing');
fea = housedata(:,1:end-1);
gnd = housedata(:,end);
fea = fea - mean(fea);
fea = fea./std(fea);
feaTrain = fea(1:400,:);
gndTrain = gnd(1:400);
feaTest = fea(401:end,:);
gndTest = gnd(401:end);

disp('Training SIGP ...');
hyp = sigp(feaTrain,gndTrain,2,'covkfn','sigp_rbf','covkpar',503.47);

disp('Mean squared error:' + string(norm(hyp.f(feaTest) - gndTest)^2/length(gndTest)));
```

In Matlab:
```
>> demoSIGPHousing
Loading housing data ...
Training SIGP ...
Mean squared error:14.2078
```

For comparison, the standard GP based on GPML Toolbox (http://www.gaussianprocess.org/gpml/code/matlab/doc/) yields a much larger mean squared error 93.1109. To verify, add GPML to the Matlab PATH, and run demoGPHousing.m. 


## References
More details can be found in our paper: https://arxiv.org/abs/1802.07528
