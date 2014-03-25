function [train_data, train_label, validation_data, ...
    validation_label, test_data, test_label] = preprocess()
% preprocess function loads the original data set, performs some preprocess
%   tasks, and output the preprocessed train, validation and test data.

% Input:
% Although this function doesn't have any input, you are required to load
% the MNIST data set from file 'mnist_all.mat'.

% Output:
% train_data: matrix of training set. Each row of train_data contains 
%   feature vector of a image
% train_label: vector of label corresponding to each image in the training
%   set
% validation_data: matrix of training set. Each row of validation_data 
%   contains feature vector of a image
% validation_label: vector of label corresponding to each image in the 
%   training set
% test_data: matrix of testing set. Each row of test_data contains 
%   feature vector of a image
% test_label: vector of label corresponding to each image in the testing
%   set

% Some suggestions for preprocessing step:
% - divide the original data set to training, validation and testing set
%       with corresponding labels
% - convert original data set from integer to double by using double()
%       function
% - normalize the data to [0, 1]
% - feature selection

load('mnist_all.mat');
train_data = vertcat(train0,train1,train2,train3,train4,train5,train6,train7,train8,train9);%dim is 500000x784
no_of_features=size(train_data,2);
% Depending on the size of individual train0, train1 matrix populate 
% the train_label matrix
%example of repmat--> B = repmat(A,4,1) means B contains 4 copies of A in the first dimension and 1 copy in the second dimension. 
label0 = [1,0,0,0,0,0,0,0,0,0];
label0 = repmat(label0,size(train0,1),1);%dim is 5923x10

label1 = [0,1,0,0,0,0,0,0,0,0];
label1= repmat(label1,size(train1,1),1);

label2 = [0,0,1,0,0,0,0,0,0,0];
label2 = repmat(label2,size(train2,1),1);

label3 = [0,0,0,1,0,0,0,0,0,0];
label3 = repmat(label3,size(train3,1),1);

label4 = [0,0,0,0,1,0,0,0,0,0];
label4 = repmat(label4,size(train4,1),1);

label5 = [0,0,0,0,0,1,0,0,0,0];
label5 = repmat(label5,size(train5,1),1);

label6 = [0,0,0,0,0,0,1,0,0,0];
label6 = repmat(label6,size(train6,1),1);

label7 = [0,0,0,0,0,0,0,1,0,0];
label7 = repmat(label7,size(train7,1),1);

label8 = [0,0,0,0,0,0,0,0,1,0];
label8 = repmat(label8,size(train8,1),1);

label9 = [0,0,0,0,0,0,0,0,0,1];
label9 = repmat(label9,size(train9,1),1);

train_label = vertcat(label0,label1,label2,label3,label4,label5,label6,label7,label8,label9);%dim is 50000x10

% Append train_label to train_data
train_data = [train_data train_label];

% Convert the train_data to double
train_data = double(train_data);

% Select random 50000 rows from train data 
total_rows = randperm(60000);
random_rows = total_rows(1:50000);
remaining_rows = setdiff(total_rows, random_rows);

% Create validation_data of 10000 rows remaining from train_data
% Also create validation_label
% Purge last column from validation_data
validation_data = train_data( remaining_rows(1:10000), :);
validation_label = validation_data(:,  no_of_features+1:end);
validation_data = validation_data(:, 1:no_of_features);

% Create train_data of 50000 rows
% Also create train_label
% Purge last column from train_data
train_data = train_data( random_rows(1:50000), :); 
train_label = train_data(:, no_of_features+1:end);
train_data = train_data(:, 1:no_of_features);

% Create test_data and test_label
test_data = vertcat(test0,test1,test2,test3,test4,test5,test6,test7,test8,test9);
test_data = double(test_data);

test_label0 = [1,0,0,0,0,0,0,0,0,0];
test_label0 = repmat(test_label0,size(test0,1),1);%dim is  980x10

test_label1 = [0,1,0,0,0,0,0,0,0,0];
test_label1= repmat(test_label1,size(test1,1),1);

test_label2 = [0,0,1,0,0,0,0,0,0,0];
test_label2 = repmat(test_label2,size(test2,1),1);

test_label3 = [0,0,0,1,0,0,0,0,0,0];
test_label3 = repmat(test_label3,size(test3,1),1);

test_label4 = [0,0,0,0,1,0,0,0,0,0];
test_label4 = repmat(test_label4,size(test4,1),1);

test_label5 = [0,0,0,0,0,1,0,0,0,0];
test_label5 = repmat(test_label5,size(test5,1),1);

test_label6 = [0,0,0,0,0,0,1,0,0,0];
test_label6 = repmat(test_label6,size(test6,1),1);

test_label7 = [0,0,0,0,0,0,0,1,0,0];
test_label7 = repmat(test_label7,size(test7,1),1);

test_label8 = [0,0,0,0,0,0,0,0,1,0];
test_label8 = repmat(test_label8,size(test8,1),1);

test_label9 = [0,0,0,0,0,0,0,0,0,1];
test_label9 = repmat(test_label9,size(test9,1),1);

test_label = vertcat(test_label0,test_label1,test_label2,test_label3,test_label4,test_label5,test_label6,test_label7,test_label8,test_label9);%dim is 10000x10


train_data = mat2gray(train_data);%Convert matrix to grayscale image
validation_data = mat2gray(validation_data);
test_data = mat2gray(test_data);

end

