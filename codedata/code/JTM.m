function [Acc,Cls,Z,A] = JTM(Xs,Xt,Ys,Yt,options)

% Transfer Joint Matching for Unsupervised Domain Adaptation.
% IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.
% Mingsheng Long, Jianmin Wang, Guiguang Ding, Jiaguang Sun, Philip S. Yu.

% Load algorithm options
if nargin < 5
    error('Algorithm parameters should be set!');
end
if ~isfield(options,'k')
    options.k = 100;
end
if ~isfield(options,'lambda')
    options.lambda = 1.0;
end
if ~isfield(options,'T')
    options.T = 10;
end
if ~isfield(options,'ker')
    options.ker = 'linear';
end
if ~isfield(options,'gamma')
    options.gamma = 1.0;
end
if ~isfield(options,'data')
    options.data = 'default';
end
k = options.k;
lambda = options.lambda;
T = options.T;
ker = options.ker;
gamma = options.gamma;
data = options.data;

fprintf('JTM:  data=%s  k=%d  lambda=%f\n',data,k,lambda);

% Set predefined variables
X = [Xs,Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
ns = size(Xs,2);
nt = size(Xt,2);
n = ns+nt;

% Construct kernel matrix
K = kernel(ker,X,[],gamma);

% Construct centering matrix
H = eye(n)-1/(n)*ones(n,n);

% Construct MMD matrix
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e*e';
M = M/norm(M,'fro');

% Transfer Joint Matching: JTM
G = speye(n);
Acc = [];
for t = 1:T
    [A,~] = eigs(K*M*K'+lambda*G,K*H*K',k,'SM');
    G(1:ns,1:ns) = diag(sparse(1./(sqrt(sum(A(1:ns,:).^2,2)+eps))));
    Z = A'*K;
    
    Cls = knnclassify(Z(:,ns+1:n)',Z(:,1:ns)',Ys,1);
    acc = sum(Cls==Yt)/nt;
    Acc = [Acc;acc(1)];
    
    fprintf('[%d]  acc=%f\n',t,full(acc(1)));
end
fprintf('Algorithm JTM terminated!!!\n\n');

end
