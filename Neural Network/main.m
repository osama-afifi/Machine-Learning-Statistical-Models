%% Osama M. Afifi
%% 07/07/2013
%% 

%  Procedure
%  ------------

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this program
input_layer_size  = 6;  
hidden_layer_size = 4;   
num_labels = 2;          
trainingRatio = 0.7;                         
						  
%% =========== Part 1: Loading Data =============
%  We start by first loading the data. 


% Load Training Data
fprintf('.................... Phase 1 .......................\n')
fprintf('Loading Data File ...\n')
Data = csvread('../Data/train2.csv');

fprintf('Setting up Feature Matrix ...\n')
feature_columns = [2, 3, 4, 5, 6, 7];
XData = Data(:,feature_columns);
m = int32(size(XData, 1)*trainingRatio);
X = XData([1:m],:);
XCV = XData([m+1:size(XData, 1)],:);

fprintf('Setting up Label Vector ...\n')
yData = Data(:,8);
y = yData([1:m]);
yCV = yData([m+1:size(yData, 1)]);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Initializing Parameters ================
%  A two layer neural network that classifies digits. we will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)
fprintf('.................... Phase 2 .......................\n')
warning('off', 'Octave:possible-matlab-short-circuit-operator');
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =================== Part 3: Training NN ===================
%  To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.

fprintf('.................... Phase 3 .......................\n')
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 150);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 4: Predict Training Acc =================
%  After training the neural network, we would like to use it to predict the labels of the training set. This lets
%  you compute the training set accuracy.

fprintf('.................... Phase 4 .......................\n')
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 5: Predict  Cross Validation =================
%  After training the neural network, we would like to use it to predict the labels of the CV set. This lets
%  you compute the CV accuracy.

fprintf('.................... Phase 5 .......................\n')
predCV = predict(Theta1, Theta2, XCV);
fprintf('\nCross Validation Set Accuracy: %f\n', mean(double(predCV == yCV)) * 100);


fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 6: Choosing best lambda =================
%% Choosing best lambda

fprintf('.................... Phase 6 .......................\n')

options = optimset('MaxIter', 50);
%  You should also try different values of lambda
lambda_values = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10];

bestPred = 0;
for lambda = lambda_values;
% Create "short hand" for the cost function to be minimized
fprintf('.................. lambda = %f..........................\n', lambda)
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1Test = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2Test = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

predCV = predict(Theta1Test, Theta2Test, XCV);
accPerc = mean(double(predCV == yCV));
fprintf('lambda = %f, Cross Validation Set Accuracy: %f\n', lambda, accPerc * 100.0);

if (bestPred<predCV);
fprintf('Best Prediction So Far\n');
bestPred = predCV;
Theta1 = Theta1Test;
Theta2 = Theta2Test;
bestlambda = lambda
endif;


endfor;
				 
%% ================= Part 7: Predict Testing Data =================
%  After training the neural network, we would like to use it to predict the labels of the tesring data

fprintf('\n.................... Phase 7 .......................\n')

fprintf('Last Training ...\n')
options = optimset('MaxIter', 500);
bestlambda = 1;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, bestlambda);
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1Test = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2Test = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

predCV = predict(Theta1Test, Theta2Test, XCV);
accPerc = mean(double(predCV == yCV));
fprintf('Best Cross Validation Set Accuracy: %f\n\n', accPerc * 100.0);

fprintf('Saving Results ... \n')


DataTest = csvread('../Data/test2.csv');
XTest = DataTest(:,feature_columns);
predTest = predict(Theta1, Theta2, XTest);
predTest( predTest==1 ) = 0; % Mapping 0 into 10
predTest( predTest==2 ) = 1; % Mapping 0 into 10
predTest = [DataTest(:,1) predTest];
csvwrite ('predTest.csv', predTest);
