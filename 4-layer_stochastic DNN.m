clear
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%Anakha V B 
%%%%%working code for MNIST digit classificatio with training done for
%%%%%60,000 samples 
%%% Deterministic with sigmoid 
% checking the validation error (55,000 for training and remaining 5,000
% for test) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

numOfLayers = 3; %%number of layers 
unitIN = 784;
unitHidd1 =256;
unitHidd2 = 128;
unitOP = 10;
eta = 0.005; 
numOfSamples =60000;
load digit.mat
load input1.mat

% to randomly pick the samples  


train = IN1(1:50000,:);
validation = IN1(50001:60000,:);

valid_digit = digit(50001:60000,:);
load theta1.mat
load theta2.mat
load theta3.mat

load desired.mat
load test1.mat %% actual test image
load test_digit.mat 
% 
X1 = horzcat(ones(50000,1),train);
Z = X1';
                                                                  
 
delta3 = zeros();
DeltaW3 = zeros();
DeltaW2 = zeros();
%DeltaW2 = zeros(unitHidd,unitIN +1);
iter = 1; 
maxiter = 30;
tic
while (iter<= maxiter)
%%activation functions 
 


  
   for i = 1:length(train)
       
        z1(:,i) = theta1*Z(:,i);
        output1(:,i) =sigmoid(z1(:,i));

        X2(:,i) = [1 output1(:,i)']';
        z2(:,i) = theta2* X2(:,i);
        output2(:,i) = sigmoid(z2(:,i));

        X3(:,i) = [1 output2(:,i)']';
        z3(:,i) = theta3* X3(:,i);
        zi = z3(:,i);

        for j = 1: unitOP
            output(j,i) = exp(zi(j))./sum(exp(zi));  %%softmax for the output layer
        end
        %%multiclass entropy as the cost fucntion

        MC(:,i) = -1 * sum(target(:,digit(i)+1).* log(output(:,i)));
%     J(iter) = -1.*sum((target(:,digit(i)+1).*log(output)+(1- expectedOutput).*log(1-output)))./numOfSamples;
    

  delta3 = (output(:,i)- target(:,digit(i)+1));
  Range_delta3 (:,:,iter)= delta3;  
  DeltaW3 = delta3 * X3(:,i)';
  
  delta2 = (theta3'*delta3).*X3(:,i).* (1- X3(:,i)); 
  delta2_new = delta2(2:end);
  Range_delta2 (:,:,iter)= delta2_new;
  DeltaW2 = delta2_new*X2(:,i)';
  
   
  back1 = X2(:,i).* (1- X2(:,i));
  delta1 = (theta2'*delta2_new).* back1;
  delta1_new = delta1(2:end);
  Range_delta1 (:,:,iter)= delta1_new;
  DeltaW1 = delta1_new * Z(:,i)';
%     eta = 0.2;
%   Delta_1 = -1.* eta.* DeltaW1;
%   Delta_2 = -1.* eta.* DeltaW2;
%   Delta_3 = -1 * eta.* DeltaW3;
  if (iter >=1 && iter <= 5)
      eta = 0.02;
      Delta_1 = -1.* eta.* DeltaW1;
      Delta_2 = -1.* eta.* DeltaW2;
      Delta_3 = -1 * eta.* DeltaW3;
  elseif (iter >5 && iter <= 15)
      eta = 0.01;
      Delta_1 = -1.* eta.* DeltaW1;
      Delta_2 = -1.* eta.* DeltaW2;
      Delta_3 = -1 * eta.* DeltaW3;
  elseif (iter >15 && iter <= 20)
      eta = 0.0055;
      Delta_1 = -1.* eta.* DeltaW1;
      Delta_2 = -1.* eta.* DeltaW2;
      Delta_3 = -1 * eta.* DeltaW3;
      
   elseif (iter >20 && iter <= 25)
      eta = 0.003;
      Delta_1 = -1.* eta.* DeltaW1;
      Delta_2 = -1.* eta.* DeltaW2;
      Delta_3 = -1 * eta.* DeltaW3;
      
   elseif (iter >25 && iter <= 30)
      eta = 0.0009;
      Delta_1 = -1.* eta.* DeltaW1;
      Delta_2 = -1.* eta.* DeltaW2;
      Delta_3 = -1 * eta.* DeltaW3;
  end
   theta1 = theta1 + Delta_1;
   theta2 = theta2 + Delta_2;
   theta3 = theta3 + Delta_3;
   Range_theta1(:,:,iter) = theta1;
   Range_theta2(:,:,iter) = theta2;
   Range_theta3(:,:,iter) = theta3;
  end
   J (iter) = sum(MC)./length(train);
   %% Accuracy check 
%    [M I] = max(output);
%    final = I-1;
%    check = final'- digit(1:numOfSamples);
%    Acc_train(iter) = length(find(check==0)).* 100./numOfSamples
   
   
   %% to determine the validation after every 50,000 images are presented  
   

   valid_set = horzcat(ones(10000,1),validation);%IN_test
   Z_valid = valid_set';
   valid_samples =length(validation);
  
   
 for ii = 1:length(validation)   
     
       z1_valid(:,ii) = theta1*Z_valid(:,ii);
       output1_valid(:,ii) =sigmoid(z1_valid(:,ii));

       X2_valid(:,ii) = [1 output1_valid(:,ii)']';
       z2_valid(:,ii) = theta2* X2_valid(:,ii);
       output2_valid(:,ii) = sigmoid(z2_valid(:,ii));

       X3_valid(:,ii) = [1 output2_valid(:,ii)']';
       z3_valid(:,ii) = theta3* X3_valid(:,ii);
       zi_valid = z3_valid(:,ii);
       
       output_valid(:,ii) = exp(zi_valid)./sum(exp(zi_valid));  %%softmax for the output layer
       
 end  
   
 [M Ivalid] = max(output_valid);
 final_valid = Ivalid-1;
 check_valid = final_valid'- valid_digit(1:length(validation));
 Acc_valid(iter) = length(find(check_valid==0)).* 100./length(validation)
   

 
 
 %%% training error   
 %% to determine the training accuracy 
 X1_train = horzcat(ones(length(train),1),train);
  Z_train = X1_train';
 alpha = 1;
 beta = 1; 

 for kk = 1:length(train)   
     
       z1_train(:,kk) = theta1*Z_train(:,kk);
       output1_train(:,kk) =sigmoid(alpha.*z1_train(:,kk));

       X2_train(:,kk) = [1 output1_train(:,kk)']';
       z2_train(:,kk) = theta2* X2_train(:,kk);
       output2_train(:,kk) = sigmoid(alpha.*z2_train(:,kk));

       X3_train(:,kk) = [1 output2_train(:,kk)']';
       z3_train(:,kk) = theta3* X3_train(:,kk);
       zi_train = z3_train(:,kk);
       
       output_train(:,kk) = exp(beta.*zi_train)./sum(exp(beta.*zi_train));  %%softmax for the output layer
       
 end  
   
 [M Itrain] = max(output_train);
 final_train = Itrain-1;
 check_train = final_train'- digit(1:length(train));
 Acc_train(iter) = length(find(check_train==0)).* 100./length(train)
 
 %% to determine the test accuracy 
 Xtest = horzcat(ones(10000,1),test);%IN_test
   Z_test = Xtest';
   test_samples =10000;
  
   
 for ii = 1:test_samples   
     
       z1_test(:,ii) = theta1*Z_test(:,ii);
       output1_test(:,ii) =sigmoid(z1_test(:,ii));

       X2_test(:,ii) = [1 output1_test(:,ii)']';
       z2_test(:,ii) = theta2* X2_test(:,ii);
       output2_test(:,ii) = sigmoid(z2_test(:,ii));

       X3_test(:,ii) = [1 output2_test(:,ii)']';
       z3_test(:,ii) = theta3* X3_test(:,ii);
       zi_test = z3_test(:,ii);
       
       output_test(:,ii) = exp(zi_test)./sum(exp(zi_test));  %%softmax for the output layer
       
 end  
   
 [M Itest] = max(output_test);
 final_test = Itest-1;
 check_test = final_test'- test_digit(1:test_samples);
 Acc_test(iter)  = length(find(check_test==0)).* 100./test_samples
 
 
 
   iter = iter +1;
   
end 









toc
figure (2)
semilogy(J,'-.r*','linewidth',2); grid on 
set(gca,'Fontsize',16);
title('Cost function');
xlabel('# Iterations');
ylabel('Average Crossentropy Error');
hold off
% save('J_zerocenter.mat','J');


figure (3) 

semilogy( 100 - Acc_test,'-.m*','linewidth',2); grid on
set(gca,'Fontsize',16);
xlabel('Epochs');
ylabel('Error (%)');
hold on 

semilogy( 100 - Acc_train,'-.r*','linewidth',2); grid on
set(gca,'Fontsize',16);
xlabel('Epochs');
ylabel('Error (%)');

hold off 
legend('Test Error','Training Error');
csvwrite('Cost_MATLAB.csv',J);
% 
% save('Acc_train_deter_30epochs_final.mat','Acc_train');
% 
% save('Acc_test_deter_30epochs_final.mat','Acc_test');
% save('Acc_valid_deter_30epochs_final.mat','Acc_valid');
% 
% save('theta1_deter_30epochs_final.mat','theta1');
% save('theta2_deter_30epochs_final.mat','theta2');
% save('theta3_deter_30epochs_final.mat','theta3');