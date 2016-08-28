function [ error ] = ComputeHiddenErr( output,out_err, weights )
% COMPUTEHIDDENERR compute the error of each neuron in hidden layer. The
% input of the function is the hidden layer output, the weights of each
% output layer neuron and the error of the output layer. The output of the 
% function is the error of the hidden layer.
%
% -output: a 1-by-n matrix and represent the output of hidden layer. n is
% the number of neurons in hidden layer.
% -out_err: a 1-by-m matrix and represent the error of output layer. m is
% the number of neurons in output layer.
% -weights: a m-by-n matrix and represent the weights value of output layer
% -error: a 1-by-n matrix and represent the error of the hidden layer.
%
% @Author: Hammer Zhang
% @Time: 2016-1-4
%
% ========================================================================

sigma = out_err * weights;
error = output .* (1 - output);
error = error .* sigma;


end

