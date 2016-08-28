function [ wHiddenLyr, wOutputLyr, bHiddenLyr, bOutputLyr, nIter,Err] = BPANN( input,Targets,nNeurons)
% ANN_MAINFRAME bulid a main frame of a ANN with only one hidden layer. The
% inputs of the function are inputs of the neural network, target labels of
% the outputs. The output of the function is the weights of each layer.
%
% -Inputs: a m-by-n matrix and represent the input of the network. n is the
% dimension of input and m is sample numbers
% -Targets: a m-by-d matrix and represent the output label of training data
% set.d is the dimension of output and m is the sample numbers
% -nNeurons: a scalar and represents the number of neurons in hidden layer
% -wHiddenLyr: a nNeurons-by-n matrix and represent the weight value of
% hidden layer.
% -wOutputLyr: a d-by-nNeurons matrix and represent the weight value of
% output layer.
%
% @Author: Hammer Zhang
% @Time: 2016-1-3
%
% =====================================================================
%% 输入预处理
% 在输入后添加一列
% Bias = ones(size(Inputs),1);
% input = [Inputs,Bias];

%% 设置学习参数
maxIter = 200;                                 % 最大迭代次数
minErr = 1e-10;                                % 最小允许误差
learningRate = 0.01;                           % 学习速率
momentum = 0.9;                                % 冲量项

%% 设置网络结构
[nSample,nInputs] = size(input);              % 输入维数
nOutputs = size(Targets,2);                   % 输出维数

% 创建隐藏层
[wHiddenLyr,bHiddenLyr] = CreateLayer(nNeurons,nInputs);
% 创建输出层
[wOutputLyr,bOutputLyr] = CreateLayer(nOutputs,nNeurons);
% 创建冲量项
predelta_wh = zeros(size(wHiddenLyr));
predelta_wo = zeros(size(wOutputLyr));
predelta_bh = zeros(size(bHiddenLyr));
predelta_bo = zeros(size(bOutputLyr));

%% 随机梯度下降迭代过程
nIter = 0;                                    % 迭代的次数
nErr = 9999.0;                         % 初始误差设为一个比较大的值，会随着迭代减小
stop = 0;                                     % 停止标志
%for k = 1 : maxIter
while nErr > minErr
    %% 判断是否终止迭代
    if stop == 1
        break;
    end
    
    %% 随机梯度下降
    for i = 1 : nSample
       %% 判断终止条件
        if nErr < minErr
            stop = 1;
            break;
        end
       %% 计算输出项
        % 计算隐藏层输出
        out_hidden = ComputeOutput(input(i,:),wHiddenLyr,bHiddenLyr);
        % 计算输出层输出
        output = ComputeOutput(out_hidden,wOutputLyr,bOutputLyr);

        % 计算误差项
        % 计算输出层误差项
        err_out = ComputeOutputErr(output,Targets(i,:));
        % 计算隐藏层误差项
        err_hidden = ComputeHiddenErr(out_hidden,err_out,wOutputLyr);

        % 更新权值
        % 更新隐藏层权值
        delta_w = learningRate * err_hidden' * input(i,:);
        wHiddenLyr = wHiddenLyr + delta_w - momentum * predelta_wh;
        % 更新隐藏层偏置值
        delta_b = learningRate * err_hidden;
        bHiddenLyr = bHiddenLyr + delta_b - momentum * predelta_bh;
        % 存储冲量项
        predelta_wh = delta_w;
        predelta_bh = delta_b;
        % 更新输出层权值
        delta_w = learningRate * err_out' * out_hidden;
        wOutputLyr = wOutputLyr + delta_w - momentum * predelta_wo;
        % 更新输出层偏置
        delta_b = learningRate * err_out;
        bOutputLyr = bOutputLyr + delta_b - momentum * predelta_bo;        
        % 存储冲量项
        predelta_wo = delta_w;
        predelta_bo = delta_b;
        %% 计算迭代终止条件
        nIter = nIter + 1;
        Err(nIter) = 0.5 * sum((output - Targets(i,:)).^2,2);
        nErr = Err(nIter);
        
        % 提个醒
        if rem(nIter,10000) == 0
            fprintf('now computing %d trials, Err is %f...\n',nIter,nErr);
    end
end
   
   

end

