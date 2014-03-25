function [obj_val obj_grad] = nnObjFunction(theta, visibleSize, hiddenSize, ...
                                    n_class, train_data,...
                                    train_label, lambda)
% nnObjFunction computes the value of objective function (negative log 
%   likelihood error function with regularization) given the parameters 
%   of Neural Networks, thetraining data, their corresponding training 
%   labels and lambda - regularization hyper-parameter.

% Input:
% theta: vector of weights of 2 matrices w1 (weights of connections from
%     input layer to hidden layer) and w2 (weights of connections from
%     hidden layer to output layer) where all of the weights are contained
%     in a single vector.
% visibleSize: number of node in input layer (not include the bias node)
% hiddenSize: number of node in hidden layer (not include the bias node)
% n_class: number of node in output layer (number of classes in
%     classification problem
% train_data: matrix of training data. Each row of this matrix
%     represents the feature vector of a particular image
% train_label: the vector of truth label of training images. Each entry
%     in the vector represents the truth label of its corresponding image.
% lambda: regularization hyper-parameter. This value is used for fixing the
%     overfitting problem.
       
% Output: 
% obj_val: a scalar value representing value of error function
% obj_grad: a SINGLE vector of gradient value of error function
% NOTE: how to compute obj_grad
% Use backpropagation algorithm to compute the gradient of error function
% for each weights in weight matrices.
% Suppose the gradient of w1 is 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reshape 'theta' vector into 2 matrices of weight w1 and w2
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit j in input 
%     layer to unit i in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit j in hidden 
%     layer to unit i in output layer.

%dim of w1 is 50x785
W1 = reshape(theta(1:hiddenSize * (visibleSize + 1)), ...
                 hiddenSize, (visibleSize + 1));
%dim of w2 is 10x51
W2 = reshape(theta((1 + (hiddenSize * (visibleSize + 1))):end), ...
                 n_class, (hiddenSize + 1));
                 
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));

a1 = train_data';%dim is 784x50000
tk = train_label';%dim is 10x50000

m = size(a1,2);% =50000

%a=activation function
%adding bias col to input layer
a1=[a1 ; repmat(1,1,size(a1,2))] ;%replicating 1 50000 times and adding it to the bottom of a1
%dim of a1 becomes 785x50000

%calculating hidden layer
z2 = W1 * a1 ;%dim of z2 is 50x50000
a2 = sigmoid(z2);%dim of a2 is 50x50000

%adding bias col to hidden layer
a2=[a2 ; repmat(1,1,size(a2,2))] ;%dim of a2 is 51x50000

z3 = W2 * a2 ;%dim of z3 is 10x50000
a3 = sigmoid(z3);%dim of a3 is 10x50000

%negative log-likelihood error function
cost=(tk.*log(a3))+((1-tk).*log(1-a3)); 
obj_val=-sum(cost(:))/m;

%calculating regularisation parameter
ew1=W1.^2;
ew2=W2.^2;
ew=((sum(ew1(:))+sum(ew2(:)))/(2*m))*lambda;

%add a regularization term into our error function to control the magnitude of parameters in Neural Network.
obj_val=obj_val+ew;

%error contribution of output layer
delta3 = (a3-tk) ;

%error contribution of output layer -1
delta2 = ( W2' * delta3 ) .* (a2 .* (1-a2));

%error partial derivatives
delta_J_W1 = ( delta2 * a1' );
delta_J_W2 = ( delta3 * a2' );

%removing bias column
delta_J_W1(size(delta_J_W1,1),:) = [] ;

%we use the gradient descent to update the weight with the following
%rule:W(new) = W(old)-(learning rate)delE(W(old))
W1grad = W1grad + delta_J_W1 ;
W2grad = W2grad + delta_J_W2 ;

%claculating gradient

%the partial derivative of new objective function with respect to weight
%from hidden layer to output layer can be calculated as follow:
W1grad = (W1grad / m )+ (lambda*W1/m);
%the partial derivative of new objective function with respect to weight from input layer to hidden
%layer can be calculated as follow:
W2grad = (W2grad / m )+ (lambda*W2/m);

obj_grad = [W1grad(:) ; W2grad(:)];

end
