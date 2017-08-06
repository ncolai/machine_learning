function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

mu = (1/m * sum(X))';

%   Un-vectorized (outer loop)
%   Remember when writing vectorized code: PREMATURE OPTIMIZATION IS THE ROOT OF ALL EVIL (as Donald Knuth would say)
%   Much better explanation of what the vectorized code is doing, since each step is laid out very clearly
%   for i=1:n
    %   mu_vector = ones(m,1) * mu(i);
    %   Un-vectorized
    %   for j=1:m
        %   sigma2(i) = sigma2(i) + 1/m * (X(j,i) - mu(i)) ^ 2;
    %   end
    %   sigma2(i) = 1/m * (X(:,i) - mu_vector)' * (X(:,i) - mu_vector);
%   end

mu_matrix = ones(m,1) * mu';
sigma2 = (1/m * sum((X - mu_matrix) .* (X - mu_matrix)))';




% =============================================================


end
