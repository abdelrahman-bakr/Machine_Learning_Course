function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


predections = X*theta ;
sqrerrors = (predections - y).^2 ;
J_UNREG = (1/(2*m))*sum(sqrerrors); 

theta_shift=theta(2:size(theta));
theta_regularization =[0;theta_shift];
Regularization_term = (lambda/(2*m))* sum((theta_regularization.^2)) ;

J= J_UNREG +Regularization_term ; 

grad(1) = (1/m)*sum((predections-y).* X(:,1));

for i = 2:size(theta,1) 
  grad(i) = ((1/m)*sum((predections-y).* X(:,i))) + (lambda/m)*theta(i) ; 
endfor












% =========================================================================

grad = grad(:);

end
