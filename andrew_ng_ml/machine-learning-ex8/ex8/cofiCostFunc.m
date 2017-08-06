function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%   Non-vectorized
%   for i=1:num_movies
    %   for j=1:num_users
        %   if R(i,j) == 1
            %   y_pred = Theta(j,:) * X(i,:)';
            %   J = J + 1/2*(y_pred - Y(i,j))^2;
        %   end
    %   end
%   end

%   Vectorized
%   Find prediction values for which R(i,j) exists, i.e. the user rated it before.
y_pred = (X * Theta') .* R;
%   Take differences between predictions and actual
y_diff = y_pred - Y;
%   Compute cost by summing squares of differences
J = 1/2 * sum(sum(y_diff .^ 2));

%   Non-vectorized
%   y_diff = y_pred - Y;
%   for i=1:num_movies
%       for j=1:num_users
%           <<<<<< Add what user j contributes to the movie gradient based on the cost function>>>>>
%           X_grad(i,:) = X_grad(i,:) + y_diff(i,j) * Theta(j,:);
%           <<<<<< Add what movie i contributes to the user gradient based on the cost function>>>>>>
%           Theta_grad(j,:) = Theta_grad(j,:) + y_diff(i,j) * X(i,:);
%       end
%   end

%   PREMATURE OPTIMIZATION IS THE ROOT OF ALL EVIL. PREMATURE OPTIMIZATION IS THE ROOT OF ALL EVIL. PREMATURE OPTIMIZATION IS THE ROOT OF ALL EVIL. PREMATURE OPTIMIZATION IS THE ROOT OF ALL EVIL. PREMATURE OPTIMIZATION IS THE ROOT OF ALL EVIL.
%   Never again will I recklessly charge into vectorization problems.
%   Semi-vectorized after wandering through the devilish maze of premature optimization.
%   for i=1:num_movies
%       <<<<<< Add all contributions from each user to movie gradient >>>>
%       X_grad(i,:) = X_grad(i,:) + y_diff(i,:) * Theta;
%   end
%   for j=1:num_users
%       <<<<<<< Add all contributions from each movie to user gradient >>>>>
%       Theta_grad(j,:) = Theta_grad(j,:) + y_diff(:,j)' * X;
%   end

%   I swear never to write my code this obtusely ever again, but I swear that this works.
%   According to Donald Knuth, this here is very devilish code because it's not very clear at all how this works.
%   I agree. It cost me a few hours trying to get something this efficient.
%   Goes to show that trying to squeeze out a little bit more performance is stupid.
X_grad = y_diff * Theta;
Theta_grad = y_diff' * X;

J = J + lambda/2 * (sum(sum(Theta .* Theta)) + sum(sum(X .* X)));
X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda * Theta;









% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
