function [salMap] = ComputeSaliency(img, paras, setting, sample_options, gt_im, min_iou)
% FUNCTION: Calculate saliency using Structured Matrix Fractorization (SMF)
% INPUT:    img - input image
%           paras - parameter setting
% OUTPUT:   saliency map

%% STEP-1. Read an input images and perform preprocessing
[img.height, img.width] = size(img.RGB(:,:,1));
% [noFrameImg.RGB, noFrameImg.frame] = RemoveFrame(img.RGB, 'sobel');
% [noFrameImg.height, noFrameImg.width, noFrameImg.channel] = size(noFrameImg.RGB);

%% STEP-2. Generate superpixels using SLIC
% [sup.label, sup.num, sup.Lab] = PerformSLIC(noFrameImg);
[sup.label, adjc_mat, pixelList] = SLIC_Split(img.RGB, 150);
sup.num = length(pixelList);
meanRgbCol = GetMeanColor(img.RGB, pixelList);
meanLabCol = colorspace('Lab<-', double(meanRgbCol)/255);
sup.Lab = meanLabCol;

% get the indexes of boundary superpixels
bndIdx = GetBndSupIdx(sup.label);
v = ismember(1:sup.num, bndIdx);

% get superpixel statistics
% sup.pixIdx = cell(sup.num, 1);
sup.pixIdx = pixelList;
sup.pixNum = zeros(sup.num,1);
for i = 1:sup.num
     temp = find(sup.label==i);
%      sup.pixIdx{i} = temp;
     sup.pixNum(i) = length(temp);
end
[sup.pos_N, sup.pos] = GetNormedMeanPos(sup.pixIdx, img.height, img.width);

%% STEP-3. Extract features
featImg = ExtractFeature(im2single(img.RGB));
for i = 1:3
    featImg(:,:,i) = mat2gray(featImg(:,:,i)).*255;
end
featMat = GetMeanFeat(featImg, sup.pixIdx);  
featMat = featMat./255;
colorFeatures = featMat(:,1:3);
medianR = median(colorFeatures(:,1)); medianG = median(colorFeatures(:,2)); medianB = median(colorFeatures(:,3));
featMat(:,1:3) = (featMat(:,1:3)-1.2*repmat([medianR, medianG, medianB],size(featMat,1),1))*1.5;
%featMapShow(mapminmax(sum(abs(featMat),2)',0,1), sup.label);

%% STEP-4. Create index tree
% get the first and second order reachable matrix (a.k.a adjacent matrix)
[fstReachMat, fstSndReachMat, ~] = GetFstSndReachMat(sup.label, sup.num);
% compute color distance between adjacent superpixels
colorDistMat = ComputeFeatDistMat(sup.Lab);

samples = Sampling(sup.num, adjc_mat, sup.pos, img.RGB, 1:sup.num, bndIdx, sample_options);
u = blPred(samples);
% imshow(CreateImageFromSPs(u, pixelList, img.height, img.width));
u = mapminmax(u,0,1);

labels = LabelSps(gt_im,pixelList,min_iou);
u = labels;
u(u<0) = 0;
u(u>0) = 1;

%% STEP-5. Get high-level priors
% load color prior matrix as used in [X. Shen and Y. Wu, CVPR'12]
if ~exist('colorPriorMat','var')
    fileID = fopen('ColorPrior','rb');
    data = fread(fileID,'double');
    fclose(fileID);
    colorPriorMat = reshape(data(end-399:end), 20, 20);
end

% get banckground prior by robust background detection [W. Zhu et al., CVPR'14]
[bgPrior, bdCon] = GetPriorByRobustBackgroundDetection(colorDistMat, fstReachMat, bndIdx);
% subplot(1,2,1);featMapShow(bgPrior, sup.label);
prior = GetHighLevelPriors(sup.pos_N, sup.num, colorPriorMat, colorFeatures, bgPrior);
% pre-processing
featMat = repmat(prior,1,53) .* featMat;


%% STEP-6. Compute affinity and laplacian matrix
% link boundary superpixels
fstSndReachMat(bndIdx, bndIdx) = 1;
fstSndReachMat_lowTri_Bnd = tril(fstSndReachMat, -1);
[tmpRow, tmpCol] = find(fstSndReachMat_lowTri_Bnd>0);
edge = [tmpRow, tmpCol];
% get the weights on edges
weightOnEdge = ComputeWeightOnEdge(colorDistMat, fstSndReachMat_lowTri_Bnd, paras.delta);
% compute affinity matrix
W = sparse([edge(:,1);edge(:,2)], [edge(:,2);edge(:,1)], [weightOnEdge; weightOnEdge], sup.num, sup.num);
DCol = full(sum(W,2));
D = spdiags(DCol,0,speye(sup.num));
M = D - W;

%% STEP-7. Structured matrix decomposition
% [L, S] = SMD(featMat', M, Tree, weight, paras.alpha, paras.beta);
[L, S] = LRSL_2(featMat', M, u, v, paras.lambda, paras.alpha, paras.beta);
sprintf('rank(L) = %d\n',rank(L))
% hist(sum(abs(S)));
S_sal = mapminmax(sum(abs(S),1),0,1);
% featMapShow(mat2gray(S_sal),sup.label);

%% STEP-8: Post-processing to get improvements
% parameters for postprocessing
if(setting.postProc)
    lambda = 0.1;  % 0.1 is good
    L_sal = lambda * (1-mapminmax(sum(abs(L),1),0,1)') + (1-lambda) * (1 - mapminmax(bgPrior',0,1)');
    S_sal = postProcessing(L_sal, S_sal, bdCon, fstReachMat, W, sup);    
end
% save saliency map
salMap = GetSaliencyMap(S_sal, sup.pixIdx, [img.height,img.width,1,img.height,1,img.width], true);

end

function samples = Sampling(sample_num, adjc_mat, mean_pos, im, idx, bd_idx, sample_options)
% Sampling from given super-pixel indices

[height,width,channels] = size(im);
samples = zeros(sample_num, sample_options.pix_per_sample, channels);

half_side = sample_options.half_side;
pix_per_sample = sample_options.pix_per_sample;
pix_per_sp = sample_options.pix_per_sp;
chosen_sps = sample_options.chosen_sps;

for i = 1:sample_num
    clear adj_idx
    % find chosen_sps centers
    adj_idx = find(adjc_mat(idx(i),:)>0);
    adj_idx(adj_idx==idx(i)) = [];
    adj_indirect = [];
    
    k_n_direct = length(adj_idx);
    for j = 1:k_n_direct
        adj_indirect = union(adj_indirect,find(adjc_mat(adj_idx(j),:)>0));
    end
    idx_direct = ismember(adj_indirect, adj_idx);
    adj_indirect = adj_indirect(~idx_direct);
    
    k_n_indirect = sample_options.adj_sps - k_n_direct;
    if k_n_indirect > length(adj_indirect)
        sprintf('No enough indirect sps... %d direct sps.', k_n_direct);
        k_n_indirect = length(adj_indirect);
    end

    bd_n = sample_options.chosen_sps - k_n_direct - k_n_indirect - 1;
    
    adj_idx = [idx(i),adj_idx];
    centers = mean_pos(adj_idx,:);
    centers = [centers; mean_pos(adj_indirect(1:k_n_indirect),:); mean_pos(bd_idx(1:bd_n),:)];
    centers = round(centers);
    centers(:,1) = max(half_side+1, centers(:,1)); centers(:,1) = min(centers(:,1), height-half_side);
    centers(:,2) = max(half_side+1, centers(:,2)); centers(:,2) = min(centers(:,2), width-half_side);

    sample = zeros(pix_per_sample,channels);
    for j = 1:chosen_sps
        cent = centers(j,:);
        area = im(cent(1)-half_side:cent(1)+half_side, cent(2)-half_side:cent(2)+half_side, :);
        area = reshape(area, [pix_per_sp, channels]);
        sample((j-1)*pix_per_sp+1:j*pix_per_sp,:) = area;
    end

    samples(i,:,:) = reshape(sample, [1,pix_per_sample,channels]);
end

end

function x = blPred(test_x)

load msra_train.mat

test_x = permute(test_x,[2 1 3]);
test_x = zscore(test_x);
test_x = permute(test_x,[2 1 3]);
HH1 = horzcat(test_x, ones(size(test_x,1),1,3));
%clear test_x;
yy1 = zeros(size(test_x,1),N2*N11);
for i = 1:N2
    beta1 = beta11{i};ps1 = ps(i);
    TT1 = mulTrans(HH1,beta1);
    TT1 = mapminmax('apply',TT1',ps1)';

    clear beta1; clear ps1;
    %yy1=[yy1 TT1];
    yy1(:,N11*(i-1)+1:N11*i) = TT1;
end
clear TT1;clear HH1;
HH2 = [yy1 .1 * ones(size(yy1,1),1)]; 
TT2 = tansig(HH2 * b2 * l2);TT3=[yy1 TT2];
clear HH2;clear b2;clear TT2;

x = TT3 * beta_;
x = 1./(1+exp(-x(:,1)));

end