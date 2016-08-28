function [ output ] = ComputeNNOut( whiddenlyr,woutputlyr,bhiddenlyr,boutputlyr,input )
% CLASSFYBYANN compute the classification result by a artifical neural
% network with given input dataset. The input of the function are the
% input weights of the hidden layer, weights of the output layer, bias of
% the hidden layer, bias of the output layer and the input data. The output
% of the function is the classify result of the network
%
% whiddenlyr: a m-by-n matrix and represent the input weights of the hidden
% layer.
% outputlyr: a d-by-m matrix and represent the input weights of the output
% layer.
% bhiddenlyr: a 1-by-m matrix and represent the input bias of the hidden
% layer.
% boutputlyr: a 1-by-d matrix and represent the input bias of the output
% layer.
% input: a s-by-n matrix and represent the input data.
% target: a s-by-n matrix and represent the target corresonding to the
% input data.
% out: a s-by-d matrix and represent the output result.
%
% @Author: Hammer Zhang
% @Time:2016-1-16
%
% ========================================================================
    % 获取输入维数
    [s,n] = size(input);

    % 计算隐藏层节点输出
    hiddenlyr_out = input * whiddenlyr';
    % 加上偏置
    bhidden = repmat(bhiddenlyr,s,1);
    hiddenlyr_out = hiddenlyr_out + bhidden;
    % 计算sigmoid单元输出
    hiddenlyr_out = 1./ (1 + exp(-hiddenlyr_out));

    % 计算输出层节点输出
    outlyr_out = hiddenlyr_out * woutputlyr';
    boutput = repmat(boutputlyr,s,1);
    output = outlyr_out + boutput;
    % 计算sigmoid单元输出
    output = 1 ./ (1 + exp(-output));

end

