function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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


test_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
test_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

size_C = size(test_C, 2);
size_sigma = size(test_sigma, 2);

test_cost = zeros(size_C * size_sigma, 3);
%fprintf("test_cost size %dx%d", size(test_cost, 1), size(test_cost, 2));

for i = 1:size_C
    for j = 1:size_sigma
        test_cost_index = (i - 1) * size_sigma + j;
        C = test_C(i);
        sigma = test_sigma(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        %model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
        predictions = svmPredict(model, Xval);
        test_cost(test_cost_index, 1) = test_C(i);
        test_cost(test_cost_index, 2) = test_sigma(j);
        test_cost(test_cost_index, 3) = mean(double(predictions ~= yval));       
    endfor
endfor

[min_val, min_index] = min(test_cost(:, 3));
%fprintf("min val=%f index=%d, min_C=%f, min_sigma=%f", min_val, min_index, test_cost(min_index, 1), test_cost(min_index, 2));
C = test_cost(min_index, 1);
sigma = test_cost(min_index, 2);

% =========================================================================

end
