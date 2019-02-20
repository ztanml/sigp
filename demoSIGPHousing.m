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