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
%h(x)
z = X * theta;
h = sigmoid(z);
% logistic h(x) and 1-h(x)
sum1 = sum(log(h),2);
sum2 = sum(log(1-h),2);

theta_reg = zeros(size(X, 2));
theta_reg = theta.^2;
theta_reg(1,:) = 0;


J = 1/m * (-y'*sum1 - (1-y)'* sum2) + lambda/(2 * m) * sum(theta_reg,1);
grad = 1/m * X' * (h-y) + lambda/m * ;

for j = 2: size(X,2),
    grad(j,:) += lambda/m * theta(j,:);
end



% =============================================================

end
