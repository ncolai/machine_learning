function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

poss_values = [0.01 0.03 0.1 0.3 1 3 10 30];
lowest_cost = 10000000;
for C_index=1:length(poss_values)
    for sigma_index=1:length(poss_values)
        model= svmTrain(X, y, poss_values(C_index), @(x1, x2) gaussianKernel(x1, x2, poss_values(sigma_index))); 
        pred = svmPredict(model, Xval);
        val_cost = 1/2 * (pred - yval)' * (pred - yval);
        if val_cost < lowest_cost
            lowest_cost = val_cost;
            C = poss_values(C_index);
            sigma = poss_values(sigma_index);
        end
    end
end

%   Alternative (model) solution: compute lowest error, i.e. fraction of cross validation examples which were classified incorrectly and use that to set C and sigma




% =========================================================================

end
