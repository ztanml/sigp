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
    'covkpar',340.1,'ykpar',0.98738,'eta',7.5899e-07);

disp("F1 score:" + num2str(F1score(sign(hyp.f(feaTest)),gndTest)));