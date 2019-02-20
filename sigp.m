function hyp = sigp(X,y,m,varargin)
% Finite-Sample Integral Gaussian Processes
% Implmentation for the paper: https://arxiv.org/abs/1802.07528
%
% Input:
%    X, y are the n-by-p feature matrix and n-by-1 label vector.
%    m specifies the rank of the desired RKHS
%
% Optional:
%    EM Parameters:
%        maxiter is the max number of EM iterations
%        tol specifies the minimum change in the per-instance log-likelihood
%           value between two consecutive EM iterations.
%
%    SDR Parameters:
%        sdr specifies the method for computing sample conditional variance
%           'slicing' - fast slicing based, 'ker' - kernel based (default)
%        order is an index array for slicing groups, same indexes will be
%           mapped to same slice, default slicing is based on y
%        ns  is the maximum number of slices, each slice corresponds to a range of y.
%           For classification, a slice contains one or more classes
%        eta is small postive number used to improve the condition of A
%        ykfn/ykpar specifies the kernel and its parameter for sdr = 'ker'
%        eta1 similar to eta, used when sdr method is 'ker' 
%
%    Mean/Covariance Functions:
%        lambda is the mean function regularization parameter
%        covkfn specifies the covariance kernel
%
% Returns the model struct hyp. Some important members are
%    hyp.f is the fitted target function f: X -> [Y,varY]
%    hyp.mf is the fitted mean function mf: X -> Y
%    hyp.nlp is a vector of negative log likelihood
%
% GitHub: https://github.com/ZilongTan/sigp
% Copyright (c) 2018-2019 Zilong Tan (zilongt@cs.cmu.edu)

hyp = struct();
[n,p] = size(X);

opt = inputParser;
opt.addParameter( 'maxiter',   20,        @(x) floor(x) > 0 );
opt.addParameter( 'tol',       1e-8,      @(x) floor(x) >= 0);
opt.addParameter( 'order',     [],        @(x) isempty(x) | length(x)==n);
opt.addParameter( 'sdr',       'ker',     @(x) strcmp(x,'slicing')|strcmp(x,'ker'));
opt.addParameter( 'ns',        0,         @(x) floor(x) > 1 & floor(x) <= n/2); % 0 for auto select
opt.addParameter( 'eta',       1e-10,     @(x) floor(x) >= 0);
opt.addParameter( 'eta1',      1e-4,      @(x) floor(x) >= 0);
opt.addParameter( 'ykfn',      @sigp_rbf, @(x) feval(x) >= 0);
opt.addParameter( 'ykpar',     std(y)/2,  @(x) true);
opt.addParameter( 'lambda',    0,         @(x) floor(x) >= 0);
opt.addParameter( 'covkfn',    @sigp_rbf, @(x) feval(x) >= 0);
opt.addParameter( 'covkpar',   1,         @(x) true);
opt.parse(varargin{:});
opt = opt.Results;    

hyp.opt = opt;

% Sort data if we choose to slice
if strcmp(opt.sdr,'slicing') && m > 0
    % Sort and slice the data by y
    % This works for both regression and classification
    if isempty(opt.order)
        [y,idx] = sort(y,'ascend');
        X = X(idx,:);
        [~,nun] = unique(y);
        hyp.Idx = idx;
    else
        y = y(opt.order);
        X = X(opt.order,:);
        [~,nun] = unique(opt.order);
        hyp.Idx = opt.order;
    end
    ns = opt.ns;
    if ns == 0 % auto select
        ns = min(length(nun),floor(n/2));
    end
    nun = [nun(2:end)-nun(1:end-1); n+1-nun(end)];
    if length(nun) <= ns
        csz = nun;
    else
        csz(1,1) = 0;
        sz = n/ns;
        i = 1;
        for j = 1:length(nun)
            if csz(i,1) >= sz
                i = i + 1;
                csz(i,1) = 0;
            end
            csz(i,1) = csz(i,1) + nun(j);
        end
    end
    ns  = length(csz); % actual number of slices
    pos = cumsum([1;csz]);
end

covkfn = @(X,Z,param) feval(opt.covkfn,X,Z,param);
CK = covkfn(X,[],opt.covkpar);

if m <= 0
    % use the full covariance kernel
    m = n;
    W = eye(n);
else
    if strcmp(opt.sdr,'slicing') 
        % Slicing-based estimation of SDR basis
        % Compute RKHS dimension reduction matrices
        A = zeros(n,n);
        for i = 1:ns
            idx = pos(i):pos(i+1)-1;
            A(idx,:) = CK(idx,:) - mean(CK(idx,:));
        end
        C = CK - mean(CK);
        A(1:n+1:end) = A(1:n+1:end) + n*opt.eta;
        [W,E] = eigs(C,A,m);
    else
        % Kernel-based estimation of SDR basis
        Ky = feval(opt.ykfn,y,[],opt.ykpar);
        Ky = Ky - mean(Ky,2);
        Ky = Ky - mean(Ky);
        V  = Ky;
        V(1:n+1:end) = V(1:n+1:end) + n*opt.eta1;
        V  = V\Ky*CK;
        V(1:n+1:end) = V(1:n+1:end) - n*opt.eta;
        LK = CK - mean(CK);
        [W,E] = eigs(LK,LK-V,m);
    end
    hyp.eigs = 1-1./diag(E);
end

% Initialize other parameters
MCK = mean(CK);
KW  = CK*W;
PiX = KW - mean(KW);
PTP = PiX'*PiX;
beta = zeros(m,1);
err = y;
s2  = opt.eta;
G   = zeros(n,n);
iSb = zeros(m,m);

WTKW = KW'*W;
efn = @(varargin) sigp_efn_cov(y,PiX,WTKW,varargin{:});

hyp.nlp = [];

% EM iterations
for i = 1:opt.maxiter
    % Update the variance components
    Sv  = inv(PTP/s2 + iSb);
    beta = Sv/s2*PiX'*err;
    Sb  = beta*beta' + Sv;
    iSb = inv(Sb);
    res = err - PiX*beta;
    s2  = s2 + (sum(res.^2) - s2^2*trace(G))/n;
    % Update the mean function
    G   = compG(PiX*pdsqrtm(Sb),s2);
    alp = efn(G,opt.lambda);
    err = efn(alp);
    % Negative log-likelihood
    nlp = (log(2*pi)*(n+m) + pdlogdet(Sb) + n*log(s2) + ...
           beta'*iSb*beta + sum((res/sqrt(s2)).^2))/2/n;
    hyp.nlp = [hyp.nlp; nlp];
    if length(hyp.nlp) > 1 && hyp.nlp(end-1) - hyp.nlp(end) < opt.tol
        break;
    end
end

hyp.Basis = W;
hyp.alpha = alp;
hyp.Pi    = PiX;
hyp.s2    = s2;

MF = W*beta;
CF = W*sqrtm(Sv);

hyp.covkfn = @(Z) covkfn(Z,X,opt.covkpar) - MCK;
hyp.mf = @(Z)-sigp_efn_cov(zeros(size(Z,1),1),hyp.covkfn(Z)*W,[],alp);
hyp.f  = @(Z) sigp_pred(hyp.covkfn(Z),hyp.mf(Z),MF,CF,s2);

end

function [pmu,pvar] = sigp_pred(KZ,muZ,MF,CF,s2)
pmu  = muZ + KZ*MF;
pvar = sum((KZ*CF).^2,2) + s2;
end

% Mean function based on covariance kernel
function val = sigp_efn_cov(y,PiX,S,G,lambda)
if nargin < 4, val = 'm+1'; return, end
if nargin == 4, val = y - PiX*G(2:end) - G(1); return, end
n  = size(G,1);
rs = sum(G);
ss = sum(rs);
PTGL= PiX'*G - PiX'*rs'/ss*rs;
val = (PTGL*PiX + n*lambda*S)\(PTGL*y);
val = [rs/ss*(y-PiX*val); val];
end

function G = compG(PiX,s2)
n = size(PiX,1);
m = size(PiX,2);
G = PiX'*PiX;
G(1:m+1:end) = G(1:m+1:end) + s2;
G = -PiX/G*PiX';
G(1:n+1:end) = G(1:n+1:end) + 1;
G = G/s2;
end

function MSQ = pdsqrtm(X)
[MSQ,S,~] = svd(X);
MSQ = MSQ.*sqrt(diag(S))'*MSQ';
end

function val = pdlogdet(X)
S = svd(X);
val = sum(log(S));
end

function K = sigp_rbf(X,Z,band)
if nargin == 0 || isempty(X), K = 1; return, end
if nargin == 3
    X = X/band;
    if ~isempty(Z)
        Z = Z/band;
    else
        Z = X;
    end
end
sqX = -sum(X.^2,2);
sqZ = -sum(Z.^2,2);
K = bsxfun(@plus, sqX, (2*X)*Z');
K = bsxfun(@plus, sqZ', K);
K = exp(K);
end
