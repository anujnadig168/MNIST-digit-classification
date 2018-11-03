close all;
clear all;
clc;

trainImg_filename = 'train-images.idx3-ubyte';
testImg_filename = 't10k-images.idx3-ubyte';
trainLbls_filename = 'train-labels.idx1-ubyte';
testLbls_filename = 't10k-labels.idx1-ubyte';

trainImg = extractImage(trainImg_filename);
testImg = extractImage(testImg_filename);
trainLbl = extractLabel(trainLbls_filename);
testLbl = extractLabel(testLbls_filename);

trainImageMt = (reshape(trainImg,size(trainImg,1) * size(trainImg,2),size(trainImg,3)));
testImageMt = (reshape(testImg,size(testImg,1) * size(testImg,2),size(testImg,3)));

n_train = length(trainLbl);
n_test = length(testLbl);

w1 = -1 + (2).*rand(15,785);
w2 = -1 + (2).*rand(10,16);

trainImageMt = [ones(1,length(trainLbl));trainImageMt];
testImageMt = [ones(1,length(testLbl));testImageMt];
y = zeros(10,n_train);
d = eye(10);
d1_train = zeros(10,n_train);
d1_test = zeros(10,n_test);

v1 = 0;
v2 = 0;

mu = 0.2;
eta = 0.001;

for i = 1:n_train
    d1_train(trainLbl(i)+1,i)=1;
end

for i = 1:n_test
    d1_test(testLbl(i)+1,i)=1;
end

flag = 1;
epoch = 0;
dist_norm_train = 0;
dist_norm_test = 0;
error_train_rate = 0;
error_test_rate = 0;

while flag == 1
    for i = 1:n_train
        v1(:,i) = w1*trainImageMt(:,i);
        y1(:,i) = [1;sigmoidfunc(v1(:,i))];
        v2(:,i) = w2*y1(:,i);
        y(:,i) = sigmoidfunc(v2(:,i));
        
        diff = (2*(d1_train(:,i) - y(:,i)));
        phi_prime = ((y(:,i) - (y(:,i).^2)));
        del1 = diff .* phi_prime;
        del2 = (w2(:,2:end)'*del1).*(((y1(2:end,i)-(y1(2:end,i).^2))));
        grad_1 = (-del1)*y1(:,i)';
        grad_2 = (-del2)*trainImageMt(:,i)';
        
        v1 = (mu*v1) - (eta.*grad_2);
        v2 = (mu*v2) - (eta.*grad_1);
        
        w1 = w1+v1;
        w2 = w2+v2;
        
        for j = 1:10
            dist(j,i) = (norm(d(:,j)-y(:,i)).^2);
        end
        
        [~,ind] = min(dist(:,i));
        pred(i) = ind-1;
        
        dist1(i) = norm(d1_train(:,i)-y(:,i)).^2;
    end
    
    dist_norm_train = [dist_norm_train mean(dist1)];
    error_train_rate = [error_train_rate ((sum(pred'~=trainLbl)./n_train)*100)];
    
    v1_test = w1*testImageMt;
    y1_test = [ones(1,n_test);sigmoidfunc(v1_test)];
    v2_test = w2*y1_test;
    y_test = sigmoidfunc(v2_test);
    
    for i = 1:n_test
        for j = 1:10
            dist_test(j,i) = norm((d(:,j)-y_test(:,i))).^2;
        end
        [~,ind_test] = min(dist_test(:,i));
        pred_test(i) = ind_test-1;
        
        dist1_test(i) = norm(d1_test(:,i)-y_test(:,i)).^2;
    end
    
    dist_norm_test = [dist_norm_test mean(dist1_test)];
    
    error_test_rate = [error_test_rate ((sum(pred_test'~=testLbl)./n_test)*100)];
    
    if error_test_rate(end) <= 7.00
        flag = 0;
    else
        flag = 1;
    end
    
    epoch = epoch + 1;
    [error_train_rate(end) error_test_rate(end)]
    [dist_norm_train(end) dist_norm_test(end)]
end

plot(1:epoch,error_train_rate(2:end));
hold on;
plot(1:epoch,error_test_rate(2:end),'r');
title('Misclassification rate');
xlabel('Epochs');
ylabel('Rate of misclassification (%)');
legend('Training misclassification rate','Testing misclassification rate'); 

figure;

plot(1:epoch,dist_norm_train(2:end));
hold on;
plot(1:epoch,dist_norm_test(2:end),'r');
title('Energy of Training and Testing data');
xlabel('Epochs');
ylabel('Energy');
legend('Training energy','Testing energy');
            
