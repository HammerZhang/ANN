function [ error ] = ComputeOutputErr( output,target )
% COMPUTEERROR compute the error of each output neuron. The inputs of the
% function are the output of the neuron and target label of each neuron.
% The output is the error of the neuron
%
% -output: a 1-by-n matrix and represents the output of the output layer. n
% is the number of output neurons
% -target: a 1-by-n matrix and represents the target label of the output
% layer.
% -error: a 1-by-n matrix and represents the error of each output neuron
%
% @Author: Hammer Zhang
% @Time:2016-1-3
%
% ========================================================================

error = output .* (1 - output);
error = error .* (target - output);

end

