function score = F1score(Pred,Act)
% Compute the F1 score for binary classification.
% Support {-1,1} and {0,1} labels.
%
% Copyright (c) 2018 Zilong Tan (ztan@cs.duke.edu)

idx = Pred == Act;

prec = sum(Act(idx)==1) / max(sum(Pred==1),1);
rec  = sum(Act(idx)==1) / max(sum(Act==1),1);

score = 2*prec*rec/max((prec + rec),1e-12);

end
