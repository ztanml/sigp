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

disp('Training GP ...');
nFea = size(feaTrain,2);
n = size(feaTrain,1);

hyp = struct();
meanfunc = {@meanSum,{@meanLinear,@meanConst}}; hyp.mean = log(ones(nFea+1,1));
covfunc = @covSEiso; hyp.cov = log(ones(2,1));
likfunc = @likGauss; hyp.lik = log(1);
% To use FITC, use the following covariance
% covfuncF = {@apxSparse,{covfunc},feaTrain(randperm(n,400),:)};
hyp = minimize(hyp, @gp, -1000, @infExact, meanfunc, covfunc, likfunc, feaTrain, gndTrain);
[ymu,ys2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, feaTrain, gndTrain, feaTest);

disp('Mean squared error:' + string(norm(ymu - gndTest)^2/length(gndTest)));
