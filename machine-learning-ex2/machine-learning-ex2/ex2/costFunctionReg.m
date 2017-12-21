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

n = size(theta);

%calculate cost
sum_cost = 0;
sum_grad = zeros(size(theta));

for i=1:m   
    sum_cost = sum_cost + ( -y(i) * log( sigmoid( X(i,:)*theta ) ) -( 1- y(i) ) * log( 1 - sigmoid( X(i,:)*theta ) ) );
end

theta_temp = theta;
theta_temp(1) = 0;

J = sum_cost / m +  lambda / m / 2 * sum( theta_temp(:).^2 );

%calculate grad
for j=1:n
    for i=1:m
        sum_grad(j) = sum_grad(j) + ( sigmoid( X(i,:)*theta ) - y(i) ) * X(i,j);
    end
end

grad(1) = sum_grad(1) / m;

for k=2:n
    grad(k) = sum_grad(k) / m + lambda / m * theta(k);
end






% =============================================================

end
