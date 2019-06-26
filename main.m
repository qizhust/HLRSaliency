clear;
warning off all;
format compact;
tic

addpath(genpath('./Dependencies'))

% prepare data
im_path = './data/SRC/';
gt_path = './data/GT/';
im_suffix = '.jpg';
gt_suffix = '.png';
imgs_list = dir(strcat(im_path,'*',im_suffix));
sample_num = length(imgs_list);

% threshold strategy
mean_th = 4;

for i = 1:sample_num
    fprintf('Processing %d image...\n',i);
    im_name = imgs_list(i).name;
    im_idx = im_name(1:strfind(im_name,'.')-1);
    img.RGB = imread(strcat(im_path,im_name));
    
    gt_name = strcat(im_idx,gt_suffix);
    gt_im = imread(strcat(gt_path,gt_name));
    [height,width] = size(gt_im);
    
    % get feature matrtix
    [featMat, M, pixel_list, bgPrior, bdCon, fstReachMat, W, sup, adj_mat, seg_im] = getFeatMat(img);
    % low-rank matrix decomposition
    [L,S,iter] = LRSL(featMat', M, 0.35, 1.1);
    
    S_l1 = sum(abs(S),1);
    S_sal_lr = mapminmax(S_l1,0,1);
    
    % coarse saliency estimation
    aa = S_sal_lr;
    pos_index = find(aa>mean_th*mean(aa));
    neg_index = find(aa<mean(aa));
    known_index = [pos_index, neg_index];
    unk_index = setdiff(1:length(aa), known_index);
    pixel_amount = sup.pixNum;

    % spatial adjacency
    for ind = 1:length(unk_index)
        adj_index = adj_mat(unk_index(ind),:);
        adj_index(unk_index(ind)) = 0;
        aa(unk_index(ind)) = sum(S_sal_lr.*adj_index.*pixel_amount)/sum(adj_index.*pixel_amount);
    end
    
    % sample selection
    idx_n = S_l1<mean(S_l1);
    im_sample_n = featMat(idx_n,:);
    idx_p = S_l1>mean_th*mean(S_l1);
    im_sample_p = featMat(idx_p,:);
    num_p = length(find(idx_p>0));
    num_n = length(find(idx_n>0));
    
    im_sample_k = vertcat(im_sample_n,im_sample_p);
    im_label_k = zeros(num_n+num_p,2);
    im_label_k(1:num_n,2) = 1;
    im_label_k(num_n+1:end,1) = 1;
    
    im_idx_k = union(find(idx_n>0), find(idx_p>0));
    im_idx_unk = setdiff(1:length(S_l1), im_idx_k);
    im_sample_unk = featMat(im_idx_unk,:);
    im_label_unk = zeros(length(im_idx_unk),2);
    im_label_unk(:,1) = aa(im_idx_unk);
    im_label_unk(:,2) = 1-im_label_unk(:,1);
    
    weights = zeros(1,length(pixel_list));
    weights(1:num_n) = 1;
    weights(num_n+1:num_n+num_p) = num_n/num_p;
    weights(num_n+num_p+1:end) = 0.5;
    
    % saliency refinement 
    train_x = [im_sample_k; im_sample_unk];
    train_y = [im_label_k; im_label_unk];
    train_x = mapminmax(train_x,0,1);
    
    C1 = 10;
    prediction = fine_proc(mapminmax(train_x,0,1), train_y, diag(weights), C1);
    prediction = prediction(num_n+num_p+1:end,:);
    pred_lb = zeros(1, length(pixel_list));
    pred_lb(im_idx_k) = S_sal_lr(im_idx_k);
    pred_lb(im_idx_unk) = prediction(:,1);
 
    lambda = 0.1;
    L_sal = lambda * (1-mapminmax(sum(abs(L),1),0,1)') + (1-lambda) * (1 - mapminmax(bgPrior',0,1)');
    S_refine = postProcessing(L_sal, pred_lb, bdCon, fstReachMat, W, sup);    
    salMap_re = GetSaliencyMap(S_refine, sup.pixIdx, [height,width,1,height,1,width], true);
    imshow(salMap_re,'border','tight')
end
