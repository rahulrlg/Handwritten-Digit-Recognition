function g = sigmoid(z)
% sigmoid computes sigmoid functoon
% Notice that z can be a scalar, a vector or a matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sigmoid computes sigmoid functoon
% Notice that z can be a scalar, a vector or a matrix

% Notice the "." operator here
g = 1 ./ (1 + exp(-z));

end
