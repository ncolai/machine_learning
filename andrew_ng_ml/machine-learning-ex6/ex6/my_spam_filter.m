%% Machine Learning Online Class
%  Exercise 6 | Spam Classification with SVMs
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
%  TODO: finish implementation of extra credit data load!!!! (Basically just switching the data source, but would be a cool idea to implement!!!!! Also will teach you a lot about how actual machine learning works insead of being a brainless addict to class assignments)

%% Initialization
clear ; close all; clc

X_new = [];

files = dir('spam');
for i=1:length(files)
    my_file = fopen(strcat('spam/', files(i).name));
    file_text = textscan('my_file', '%s');
    disp(file_text);
    word_indices = processEmail(file_text);
    features = emailFeatures(word_indices);
    disp(word_indices);
    disp(features);
    fclose(my_file);
end
