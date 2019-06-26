function [L, S, iter] = LRSL_2( F, M, lambda, alpha)
%% intialize
tol1 = 1e-5; %threshold for the error in constraint
tol2 = 1e-5; %threshold for the change in the solutions
maxIter = 1000;
rho = 1.1;
mu = 1e-1;
max_mu = 1e10;
[d, n] = size(F);

% to save time
normfF = norm(F,'fro');

% initialize optimization variables
L = zeros(d,n);
Z1 = zeros(d,n);
Z2 = zeros(d,n);
S = zeros(d,n);
Y1 = zeros(d,n);
Y2 = zeros(d,n);
Y3 = zeros(d,n);
svp = 5; % for svd

%% start main loop
iter = 0;
%disp(['initial rank=' num2str(rank(Z))]);
while iter < maxIter
    iter = iter + 1;
    
    % copy Z and S to compute the change in the solutions
    Lk = L;
    Sk = S;
    % to save time
    Y1_mu = Y1./mu;
    Y2_mu = Y2./mu;
    Y3_mu = Y3./mu;
    
    % update L  
    temp = F - S + Y1_mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    L = U(:,1:svp) * diag(sigma) * V(:,1:svp)';
  
    % update Z1
    Z1 = Y2_mu + S;
    
    % udpate Z2
    temp = (2*alpha).*M + mu.*eye(n);
    Z2 = (mu.*S + Y3) / temp; %faster and more accurate than inv(temp)
    
    % update S    
    T = 1/3.* ( F - L + Z1 + Z2 + Y1_mu - Y2_mu - Y3_mu);
   
    S = T;
    S_l1 = sum(abs(S));
    tmp = lambda/(3*mu);
    for j = 1:n
        if S_l1(j) > tmp
            S(:,j) = (S_l1(j)-tmp)/S_l1(j).*S(:,j);
        else
            S(:,j) = 0;
        end
    end

    % check convergence condition
    leq1 = F - L - S;
    leq2 = S - Z1;
    leq3 = S - Z2;
    relChgL = norm(L - Lk,'fro')/normfF;
    relChgS = norm(S - Sk,'fro')/normfF;
    relChg = max(relChgL, relChgS);
    recErr = norm(leq1,'fro')/normfF; 
    
    convergenced = recErr <tol1 && relChg < tol2;
    
    if iter==1 || mod(iter,50)==0 || convergenced
        %disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
        %    ',rank=' num2str(rank(L,1e-3*norm(L,2))) ',stopADM=' num2str(recErr,'%2.3e')]);
    end
    if convergenced
%         sprintf('iter %d, converged...', iter)
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        mu = min(max_mu,mu*rho);        
    end
end
