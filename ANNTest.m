%% 该代码为基于BP网络的语言识别

%% 清空环境变量
clc
clear

%% 训练数据预测数据提取及归一化

%下载四类语音信号
load data1 c1
load data2 c2
load data3 c3
load data4 c4

%四个特征信号矩阵合成一个矩阵
data(1:500,:)=c1(1:500,:);
data(501:1000,:)=c2(1:500,:);
data(1001:1500,:)=c3(1:500,:);
data(1501:2000,:)=c4(1:500,:);

%从1到2000间随机排序
k=rand(1,2000);
[m,n]=sort(k);

%输入输出数据
input=data(:,2:25);
output1 =data(:,1);

%把输出从1维变成4维
output=zeros(2000,4);
for i=1:2000
    switch output1(i)
        case 1
            output(i,:)=[1 0 0 0];
        case 2
            output(i,:)=[0 1 0 0];
        case 3
            output(i,:)=[0 0 1 0];
        case 4
            output(i,:)=[0 0 0 1];
    end
end

%随机提取1500个样本为训练样本，500个样本为预测样本
input_train=input(n(1:1500),:)';
output_train=output(n(1:1500),:)';
input_test=input(n(1501:2000),:)';
output_test=output(n(1501:2000),:)';

%输入数据归一化
[inputn,inputps]=mapminmax(input_train);

%% 训练神经网络
[whiddenlyr,woutputlyr,bhiddenlyr,boutputlyr,iternum] = BPANN(inputn',output_train',25);

%% 语音信号特征分类
inputn_test=mapminmax('apply',input_test,inputps);
inputn_test = inputn_test';
fore=zeros(500,4);
for i = 1 : 500
    % 计算隐藏层输出
    hidden_sum = inputn_test(i,:) * whiddenlyr' + bhiddenlyr;
    denh = ones(1,25);
    hidden_out = denh ./ (1 + exp(-hidden_sum));
    
    % 计算输出层输出
    out_sum = hidden_out * woutputlyr' + boutputlyr;
    den = ones(1,4);
    fore(i,:) = den ./ (1 + exp(-out_sum));
end

%% 结果分析
%根据网络输出找出数据属于哪类
output_fore=zeros(500,1);
for i=1:500
    [o,output_fore(i,1)]=max(fore(i,:));
end

%BP网络预测误差
output_t = output1(n(1501:2000));
error=output_fore-output_t;

%画出预测语音种类和实际语音种类的分类图
figure(1)
plot(output_fore,'r')
hold on
plot(output1(n(1501:2000)),'b')
legend('预测语音类别','实际语音类别')

%画出误差图
figure(2)
plot(error)
title('BP网络分类误差','fontsize',12)
xlabel('语音信号','fontsize',12)
ylabel('分类误差','fontsize',12)


k=zeros(1,4);  
%找出判断错误的分类属于哪一类
for i=1:500
    if error(i)~=0
        [b,c]=max(output_test(:,i));
        switch c
            case 1 
                k(1)=k(1)+1;
            case 2 
                k(2)=k(2)+1;
            case 3 
                k(3)=k(3)+1;
            case 4 
                k(4)=k(4)+1;
        end
    end
end

%找出每类的个体和
kk=zeros(1,4);
for i=1:500
    [b,c]=max(output_test(:,i));
    switch c
        case 1
            kk(1)=kk(1)+1;
        case 2
            kk(2)=kk(2)+1;
        case 3
            kk(3)=kk(3)+1;
        case 4
            kk(4)=kk(4)+1;
    end
end

%正确率
rightridio=(kk-k)./kk;
disp('正确率')
disp(rightridio);



















