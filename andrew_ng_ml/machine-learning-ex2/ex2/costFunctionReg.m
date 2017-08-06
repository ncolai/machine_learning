function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
h = sigmoid(z);
[t_h, t_w] = size(theta);
no_theta1 = theta(2:t_h);
J = -1/m * sum(y'*log(h) + (1-y)'*log(1-h)) + 1/(2*m) * lambda * sum(no_theta1' * no_theta1);
grad = 1/m * (h - y)' * X;
grad(2) = grad(2) + 1/m * lambda * theta(2);
grad(3) = grad(3) + 1/m * lambda * theta(3);
% TODO: implement a vectorized version of this...


% =============================================================

end
