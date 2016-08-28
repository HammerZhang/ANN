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
%% ����Ԥ����
% ����������һ��
% Bias = ones(size(Inputs),1);
% input = [Inputs,Bias];

%% ����ѧϰ����
maxIter = 200;                                 % ����������
minErr = 1e-10;                                % ��С�������
learningRate = 0.01;                           % ѧϰ����
momentum = 0.9;                                % ������

%% ��������ṹ
[nSample,nInputs] = size(input);              % ����ά��
nOutputs = size(Targets,2);                   % ���ά��

% �������ز�
[wHiddenLyr,bHiddenLyr] = CreateLayer(nNeurons,nInputs);
% ���������
[wOutputLyr,bOutputLyr] = CreateLayer(nOutputs,nNeurons);
% ����������
predelta_wh = zeros(size(wHiddenLyr));
predelta_wo = zeros(size(wOutputLyr));
predelta_bh = zeros(size(bHiddenLyr));
predelta_bo = zeros(size(bOutputLyr));

%% ����ݶ��½���������
nIter = 0;                                    % �����Ĵ���
nErr = 9999.0;                         % ��ʼ�����Ϊһ���Ƚϴ��ֵ�������ŵ�����С
stop = 0;                                     % ֹͣ��־
%for k = 1 : maxIter
while nErr > minErr
    %% �ж��Ƿ���ֹ����
    if stop == 1
        break;
    end
    
    %% ����ݶ��½�
    for i = 1 : nSample
       %% �ж���ֹ����
        if nErr < minErr
            stop = 1;
            break;
        end
       %% ���������
        % �������ز����
        out_hidden = ComputeOutput(input(i,:),wHiddenLyr,bHiddenLyr);
        % ������������
        output = ComputeOutput(out_hidden,wOutputLyr,bOutputLyr);

        % ���������
        % ��������������
        err_out = ComputeOutputErr(output,Targets(i,:));
        % �������ز������
        err_hidden = ComputeHiddenErr(out_hidden,err_out,wOutputLyr);

        % ����Ȩֵ
        % �������ز�Ȩֵ
        delta_w = learningRate * err_hidden' * input(i,:);
        wHiddenLyr = wHiddenLyr + delta_w - momentum * predelta_wh;
        % �������ز�ƫ��ֵ
        delta_b = learningRate * err_hidden;
        bHiddenLyr = bHiddenLyr + delta_b - momentum * predelta_bh;
        % �洢������
        predelta_wh = delta_w;
        predelta_bh = delta_b;
        % ���������Ȩֵ
        delta_w = learningRate * err_out' * out_hidden;
        wOutputLyr = wOutputLyr + delta_w - momentum * predelta_wo;
        % ���������ƫ��
        delta_b = learningRate * err_out;
        bOutputLyr = bOutputLyr + delta_b - momentum * predelta_bo;        
        % �洢������
        predelta_wo = delta_w;
        predelta_bo = delta_b;
        %% ���������ֹ����
        nIter = nIter + 1;
        Err(nIter) = 0.5 * sum((output - Targets(i,:)).^2,2);
        nErr = Err(nIter);
        
        % �����
        if rem(nIter,10000) == 0
            fprintf('now computing %d trials, Err is %f...\n',nIter,nErr);
    end
end
   
   

end

