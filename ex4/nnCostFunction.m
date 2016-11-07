function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% calculate h(x)
A_1 = [ones(m, 1) X];
Z_2 = A_1 * Theta1';
A_2 = sigmoid(Z_2);
A_2 = [ones(size(A_2, 1), 1) A_2];
hx = sigmoid(A_2 * Theta2');
[hx_max, hax_max_i] = max(hx, [], 2);

%fprintf('\nZ_2 size=%d x %d \n', size(Z_2, 1), size(Z_2, 2));
%fprintf('\nh(x) size=%d x %d \n', size(hx, 1), size(hx, 2));

% construct y_ext which convert the y into the [1;0;0;...;0] pattern
% caculate the J
y_ext = zeros(size(hx))';
for i = 1:m
    y_ext(y(i), i) = 1;
    J = J + ((-1) * log(hx(i, :)) * y_ext(:, i) - log(1 - hx(i, :)) * (1 - y_ext(:, i)));
    %    fprintf('\n J = %f\n', J);
endfor
J = J / m;
%fprintf('\ny_ext size=%d x %d \n', size(y_ext, 1), size(y_ext, 2));

%calculate the regulized J
Theta1_t = Theta1;
Theta1_t(:, 1) = zeros(size(Theta1_t, 1), 1);
Theta2_t = Theta2;
Theta2_t(:, 1) = zeros(size(Theta2_t, 1), 1);
J = J + lambda / (2 * m) * (sum(sum(Theta1_t .^ 2)) + sum(sum(Theta2_t .^ 2)));

%calculate backpropagation
delta_3 = zeros(size(hx))';
delta_2 = zeros(size(Theta2, 2), size(hx, 2));
A_3 = hx';
%fprintf('\ndelta_3 size=%d x %d \n', size(delta_3, 1), size(delta_3, 2));
%1. Calculate delta at output layer (here k=3)
for i = 1:m
    delta_3(:, i) = A_3(:, i) - y_ext(:, i);
endfor
delta_2 = Theta2' * delta_3;
%fprintf('\ndelta_2 size=%d x %d \n', size(delta_2, 1), size(delta_2, 2));
g_inv_z2 = (A_2 .* (1 - A_2))';
delta_2 = delta_2 .* g_inv_z2;
delta_2 = delta_2(2:end, :);
%fprintf('\ndelta_2 size=%d x %d \n', size(delta_2, 1), size(delta_2, 2));

%fprintf('\nTheta1_grad size=%d x %d \n', size(Theta1_grad, 1), size(Theta1_grad, 2));
%fprintf('\nTheta2_grad size=%d x %d \n', size(Theta2_grad, 1), size(Theta2_grad, 2));
for i = 1:m
    Theta2_grad = Theta2_grad +  delta_3(:, i) * A_2(i, :);
    Theta1_grad = Theta1_grad +  delta_2(:, i) * A_1(i, :);
endfor
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

%regulized gradients
Theta1_grad = Theta1_grad + lambda / m * Theta1_t;
Theta2_grad = Theta2_grad + lambda / m * Theta2_t;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
