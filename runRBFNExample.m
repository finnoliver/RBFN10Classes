% ======== runRBFNExample ========
% This script trains an RBF Network on an example dataset, and plots the
% resulting score function and decision boundary.
% 
% There are three main steps to the training process:
%   1. Prototype selection through k-means clustering.
%   2. Calculation of beta coefficient (which controls the width of the 
%      RBF neuron activation function) for each RBF neuron.
%   3. Training of output weights for each category using gradient descent.
%
% Once the RBFN has been trained this script performs the following:
%   1. Generates a contour plot showing the output of the category 1 output
%      node.
%   2. Shows the original dataset with the placements of the protoypes and
%      an approximation of the decision boundary between the two classes.
%   3. Evaluates the RBFN's accuracy on the training set.

% $Author: ChrisMcCormick $    $Date: 2014/08/18 22:00:00 $    $Revision: 1.3 $

% Clear all existing variables from the workspace.
clear;
close all;

% Add the subdirectories to the path.
addpath('kMeans');
addpath('RBFN');

% Load the data set. 
% This loads two variables, X and y.
%   X - The dataset, 1 sample per row.
%   y - The corresponding label
% The data is randomly sorted and grouped by category.
data = load('dataset.csv');

X = data(:, 1:784);
y = data(:, 785);

y1 = ones(35000,1);
y1 = y1+y;

% Set 'm' to the number of data points.
m = size(X, 1);

% ===================================
%     Train RBF Network
% ===================================

disp('Training the RBFN...');

% Train the RBFN using 10 centers per category.
[Centers, betas, Theta] = trainRBFN(X, y1, 10, true);
 
% ================================
%         Contour Plots
% ================================

disp('Evaluating RBFN over input space...');

% Define a grid over which to evaluate the RBFN.
gridSize = 500;
u = linspace(-10, 10, gridSize);
v = linspace(-10, 10, gridSize);

% We'll store the scores for each category as well as the 'prediction' for
% each point on the grid.


scores1 = zeros(length(u), length(v));
scores2 = zeros(length(u), length(v));
scores3 = zeros(length(u), length(v));
scores4 = zeros(length(u), length(v));
scores5 = zeros(length(u), length(v));
scores6 = zeros(length(u), length(v));
scores7 = zeros(length(u), length(v));
scores8 = zeros(length(u), length(v));
scores9 = zeros(length(u), length(v));
scores10 = zeros(length(u), length(v));
p = zeros(length(u), length(v));

% Evaluate the RBFN over the grid.
% For each row of the grid...
for (i = 1 : length(u))
    
    % Report our progress every 10th row.
    if (mod(i, 10) == 0)
        fprintf('  Grid row = %d / %d...\n', i, gridSize);
        if exist('OCTAVE_VERSION') fflush(stdout); end;
    end
    
    % For each column of the grid...
    for (j = 1 : length(v))
        
        % Compute the categorys
		scores = evaluateRBFN(Centers, betas, Theta, [u(i), v(j)]);
        
		scores1(i, j) = scores(1, 1);
        scores2(i, j) = scores(2, 1);
     	scores3(i, j) = scores(3, 1);
        scores4(i, j) = scores(4, 1);
        scores5(i, j) = scores(5, 1);
        scores6(i, j) = scores(6, 1);
        scores7(i, j) = scores(7, 1);
     	scores8(i, j) = scores(8, 1);
        scores9(i, j) = scores(9, 1);
        scores10(i, j) = scores(10, 1);
        
        % Pick the higher score.
        if (scores1(i, j) == scores2(i, j))
            p(i,j) = 1.5;
        elseif (scores1(i, j) > scores2(i, j) && scores1(i, j)  > scores3(i, j) && scores1(i, j)  > scores4(i, j) && scores1(i, j) > scores5(i, j) && scores1(i, j) > scores6(i, j) && scores1(i, j)  > scores7(i, j) && scores1(i, j)  > scores8(i, j) && scores1(i, j) > scores9(i, j)&& scores1(i, j) > scores10(i, j))
            p(i, j) = 1;
        elseif (scores2(i, j) > scores1(i, j) && scores2(i, j)  > scores3(i, j) && scores2(i, j)  > scores4(i, j) && scores2(i, j) > scores5(i, j) && scores2(i, j) > scores6(i, j) && scores2(i, j)  > scores7(i, j) && scores2(i, j)  > scores8(i, j) && scores2(i, j) > scores9(i, j)&& scores2(i, j) > scores10(i, j))
            p(i, j) = 2;
        elseif (scores3(i, j) > scores1(i, j) && scores3(i, j)  > scores2(i, j) && scores3(i, j)  > scores4(i, j) && scores3(i, j) > scores5(i, j) && scores3(i, j) > scores6(i, j) && scores3(i, j)  > scores7(i, j) && scores3(i, j)  > scores8(i, j) && scores3(i, j) > scores9(i, j) && scores2(i, j) > scores10(i, j))
            p(i, j) = 3;
        elseif (scores4(i, j) > scores1(i, j) && scores4(i, j)  > scores2(i, j) && scores4(i, j)  > scores3(i, j) && scores4(i, j) > scores5(i, j) && scores4(i, j) > scores6(i, j) && scores4(i, j)  > scores7(i, j) && scores4(i, j)  > scores8(i, j) && scores4(i, j) > scores9(i, j) && scores4(i, j) > scores10(i, j))
            p(i, j) = 4;
        elseif (scores5(i, j) > scores1(i, j) && scores5(i, j)  > scores2(i, j) && scores5(i, j)  > scores3(i, j) && scores5(i, j) > scores4(i, j) && scores5(i, j) > scores6(i, j) && scores5(i, j)  > scores7(i, j) && scores5(i, j)  > scores8(i, j) && scores5(i, j) > scores9(i, j) && scores5(i, j) > scores10(i, j))
            p(i, j) = 5;
        elseif (scores6(i, j) > scores1(i, j) && scores6(i, j)  > scores2(i, j) && scores6(i, j)  > scores3(i, j) && scores6(i, j) > scores4(i, j) && scores6(i, j) > scores5(i, j) && scores6(i, j)  > scores7(i, j) && scores6(i, j)  > scores8(i, j) && scores6(i, j) > scores9(i, j) && scores6(i, j) > scores10(i, j))
            p(i, j) = 6;
        elseif (scores7(i, j) > scores1(i, j) && scores7(i, j)  > scores2(i, j) && scores7(i, j)  > scores3(i, j) && scores7(i, j) > scores4(i, j) && scores7(i, j) > scores6(i, j) && scores7(i, j)  > scores5(i, j) && scores7(i, j)  > scores8(i, j) && scores7(i, j) > scores9(i, j) && scores7(i, j) > scores10(i, j))
            p(i, j) = 7;
        elseif (scores8(i, j) > scores1(i, j) && scores8(i, j)  > scores2(i, j) && scores8(i, j)  > scores3(i, j) && scores8(i, j) > scores4(i, j) && scores8(i, j) > scores6(i, j) && scores8(i, j)  > scores7(i, j) && scores8(i, j)  > scores5(i, j) && scores8(i, j) > scores9(i, j) && scores8(i, j) > scores10(i, j))
            p(i, j) = 8;
        elseif (scores9(i, j) > scores1(i, j) && scores9(i, j)  > scores2(i, j) && scores9(i, j)  > scores3(i, j) && scores9(i, j) > scores4(i, j) && scores9(i, j) > scores6(i, j) && scores9(i, j)  > scores7(i, j) && scores9(i, j)  > scores8(i, j) && scores9(i, j) > scores5(i, j) && scores9(i, j) > scores10(i, j))
            p(i, j) = 9;
        elseif (scores10(i, j) > scores1(i, j) && scores10(i, j)  > scores2(i, j) && scores10(i, j)  > scores3(i, j) && scores10(i, j) > scores4(i, j) && scores10(i, j) > scores6(i, j) && scores10(i, j)  > scores7(i, j) && scores10(i, j)  > scores8(i, j) && scores10(i, j) > scores9(i, j) && scores10(i, j) > scores5(i, j))
            p(i, j) = 10;
        else 
            p(i, j) = 0;
        end
    end
end

fprintf('Minimum category 1 score: %.2f\n', min(min(scores1)));
fprintf('Maximum category 1 score: %.2f\n', max(max(scores1)));
fprintf('Minimum category 2 score: %.2f\n', min(min(scores2)));
fprintf('Maximum category 2 score: %.2f\n', max(max(scores2)));
fprintf('Minimum category 3 score: %.2f\n', min(min(scores3)));
fprintf('Maximum category 3 score: %.2f\n', max(max(scores3)));
fprintf('Minimum category 4 score: %.2f\n', min(min(scores4)));
fprintf('Maximum category 4 score: %.2f\n', max(max(scores4)));
fprintf('Minimum category 5 score: %.2f\n', min(min(scores5)));
fprintf('Maximum category 5 score: %.2f\n', max(max(scores5)));
fprintf('Minimum category 6 score: %.2f\n', min(min(scores6)));
fprintf('Maximum category 6 score: %.2f\n', max(max(scores6)));
fprintf('Minimum category 7 score: %.2f\n', min(min(scores7)));
fprintf('Maximum category 7 score: %.2f\n', max(max(scores7)));
fprintf('Minimum category 8 score: %.2f\n', min(min(scores8)));
fprintf('Maximum category 8 score: %.2f\n', max(max(scores8)));
fprintf('Minimum category 9 score: %.2f\n', min(min(scores9)));
fprintf('Maximum category 9 score: %.2f\n', max(max(scores9)));
fprintf('Minimum category 10 score: %.2f\n', min(min(scores10)));
fprintf('Maximum category 10 score: %.2f\n', max(max(scores10)));

if exist('OCTAVE_VERSION') fflush(stdout); end;

% ========================================
%       Measure Training Accuracy
% ========================================

disp('Measuring training accuracy...');

numRight = 0;

wrong = [];

% For each training sample...
for (i = 1 : m)
    % Compute the scores for both categories.
    scores = evaluateRBFN(Centers, betas, Theta, X(i, :));
    
	[maxScore, category] = max(scores);
	
    % Validate the result.
    if (category == y1(i))
        numRight = numRight + 1;
    else
        wrong = [wrong; X(i, :)];
    end
    
end

% Mark the incorrectly recognized samples with a black asterisk.
%plot(wrong(:, 1), wrong(:, 2), 'k*');

accuracy = numRight / m * 100;
fprintf('Training accuracy: %d / %d, %.1f%%\n', numRight, m, accuracy);
if exist('OCTAVE_VERSION') fflush(stdout); end;
