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

% Fit the Gaussian process
nFea = size(feaTrain,2);
n = size(feaTrain,1);

disp("Training a GP ...");
hyp = struct();
meanfunc = {@meanSum,{@meanLinear,@meanConst}}; hyp.mean = log(ones(nFea+1,1));
covfunc = @covSEiso; hyp.cov = log(ones(2,1));
% Use the following line for FITC
% covfuncF = {@apxSparse,{covfunc},feaTrain(randperm(n,n/2),:)};
likfunc = @likGauss; hyp.lik = 0;
hyp = minimize(hyp, @gp, -1000, @infLaplace, meanfunc, covfunc, likfunc, feaTrain, gndTrain);

[ymu,ys2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, feaTrain, gndTrain, feaTest);
disp("F1 score:" +num2str(F1score(sign(ymu),gndTest)));
