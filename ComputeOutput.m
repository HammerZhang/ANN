function [ Activation ] = ComputeOutput( Input, Weights,Bias)
% COMPUTEOUTPUT compute the outputs of a neuron layer. The inputs of the
% function are the input of matrix of the neuron layer and the weights of
% inputs, the output of the function is activation value of each neuron.
%
% - Input: a 1-by-m matrix and represent the input of the layer. m is the
% number of the inputs
% -Weights: a n-by-m matrix and represent the weights of inputs. n is the
% number of neurons in the next layer
% -Bias: a 1-by-n matrix and represent the bias
% -Activation: a 1-by-n matrix and represent the activation value of the
% layer. n is the number of neurons in this layer.
%
% @Author:Hammer Zhang
% @Time:2016-1-3
%
% =======================================================================

InputSum = Input * Weights' + Bias;
Activation = Sigmoid(InputSum);

    function [inActivation] = Sigmoid(inInput)
        inActivation = 1 ./ (1 + exp(-inInput));
    end

end

