%% �ô���Ϊ����BP���������ʶ��

%% ��ջ�������
clc
clear

%% ѵ������Ԥ��������ȡ����һ��

%�������������ź�
load data1 c1
load data2 c2
load data3 c3
load data4 c4

%�ĸ������źž���ϳ�һ������
data(1:500,:)=c1(1:500,:);
data(501:1000,:)=c2(1:500,:);
data(1001:1500,:)=c3(1:500,:);
data(1501:2000,:)=c4(1:500,:);

%��1��2000���������
k=rand(1,2000);
[m,n]=sort(k);

%�����������
input=data(:,2:25);
output1 =data(:,1);

%�������1ά���4ά
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

%�����ȡ1500������Ϊѵ��������500������ΪԤ������
input_train=input(n(1:1500),:)';
output_train=output(n(1:1500),:)';
input_test=input(n(1501:2000),:)';
output_test=output(n(1501:2000),:)';

%�������ݹ�һ��
[inputn,inputps]=mapminmax(input_train);

%% ѵ��������
[whiddenlyr,woutputlyr,bhiddenlyr,boutputlyr,iternum] = BPANN(inputn',output_train',25);

%% �����ź���������
inputn_test=mapminmax('apply',input_test,inputps);
inputn_test = inputn_test';
fore=zeros(500,4);
for i = 1 : 500
    % �������ز����
    hidden_sum = inputn_test(i,:) * whiddenlyr' + bhiddenlyr;
    denh = ones(1,25);
    hidden_out = denh ./ (1 + exp(-hidden_sum));
    
    % ������������
    out_sum = hidden_out * woutputlyr' + boutputlyr;
    den = ones(1,4);
    fore(i,:) = den ./ (1 + exp(-out_sum));
end

%% �������
%������������ҳ�������������
output_fore=zeros(500,1);
for i=1:500
    [o,output_fore(i,1)]=max(fore(i,:));
end

%BP����Ԥ�����
output_t = output1(n(1501:2000));
error=output_fore-output_t;

%����Ԥ�����������ʵ����������ķ���ͼ
figure(1)
plot(output_fore,'r')
hold on
plot(output1(n(1501:2000)),'b')
legend('Ԥ���������','ʵ���������')

%�������ͼ
figure(2)
plot(error)
title('BP����������','fontsize',12)
xlabel('�����ź�','fontsize',12)
ylabel('�������','fontsize',12)


k=zeros(1,4);  
%�ҳ��жϴ���ķ���������һ��
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

%�ҳ�ÿ��ĸ����
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

%��ȷ��
rightridio=(kk-k)./kk;
disp('��ȷ��')
disp(rightridio);



















