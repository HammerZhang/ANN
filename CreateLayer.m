function [ Weights, Bias ] = CreateLayer( nNeurons,nInputs )
% CREATELAYER create the hidden layer or output layer at given number of
% neurons and given number of inputs of each neuron. The inputs of the 
% function are number of neurons and number of inputs of each neuron.
% The outputs of the function are weight matrix of the layer and the number 
% of neurons inthe layer. The weight matrix are initilized with small 
% random numbers
%
% -nNeurons: a scalar and represents the number of neurons in this layer
% -nInputs: a scalar and represents the number of inputs of each neuron
% -Weights: a nNeurons-by-m matrix and represents the values of
% weights of the layer
% -nNeuronsPerlyr: a scalar and represents the number of neurons in each
% layer
%
% @Author: Hammer Zhang
% @Time: 2016-1-3
%
% =======================================================================

Weights = 1/20 * rands(nNeurons,nInputs);
Bias = zeros(1,nNeurons);

end

