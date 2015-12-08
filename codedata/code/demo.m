% Transfer Joint Matching for Unsupervised Domain Adaptation.
% IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.
% Mingsheng Long, Jianmin Wang, Guiguang Ding, Jiaguang Sun, Philip S. Yu.

clear all;

% Set algorithm parameters
options.k = 20;             % #subspace bases, default=20
options.lambda = 1.0;       % regularizer, default = 1.0
options.T = 10;             % #iterations, default=10
options.ker = 'linear';     % kernel type, default='linear'

srcStr = {'Caltech10','Caltech10','Caltech10','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgtStr = {'amazon','webcam','dslr','Caltech10','webcam','dslr','Caltech10','amazon','dslr','Caltech10','amazon','webcam'};
result = [];
for iData = 1:12
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    % Preprocess data using Z-score
    load(['../data/' src '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
    Xs = zscore(fts,1)';
    Ys = labels;
    load(['../data/' tgt '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
    Xt = zscore(fts,1)';
    Yt = labels;
    
    Cls = knnclassify(Xt',Xs',Ys,1);
    Acc = sum(Cls==Yt)/length(Yt);
    fprintf('acc=%0.4f\n',full(Acc));
    
    [Acc,Cls,Z,A] = JTM(Xs,Xt,Ys,Yt,options);
    result = [result;Acc(end)];
end
