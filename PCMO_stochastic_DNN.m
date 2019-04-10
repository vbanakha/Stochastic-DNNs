

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Aim: 4 layer stochastic network for MNIST classification with 
% back propagation. On - Off ratio = 1.8 and 26 levels , Nbits = 10 
%

% Author : Anakha V B 
%  run for different reset frequency
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 
clear
close all
clc

disp(' 100, sigma = PCMO, x');

s = RandStream('mt19937ar','Seed',52);
          RandStream.setGlobalStream(s);
numOfLayers = 3; %%number of layers 
unitIN = 784;
unitHidd1 =256;
unitHidd2 = 128;
unitOP = 10;
numOfSamples =60000;
load ./../digit.mat
load ./../input1.mat
load ./../theta1_26_SD.mat
load ./../theta2_26_SD.mat
load ./../theta3_26_SD.mat
load ./../desired.mat

load ./../Gp1_26_SD.mat
load ./../Gp2_26_SD.mat
load ./../Gp3_26_SD.mat
load ./../Gn1_26_SD.mat
load ./../Gn2_26_SD.mat
load ./../Gn3_26_SD.mat

countn3 = zeros(unitOP, unitHidd2 +1);
countp3 = zeros(unitOP, unitHidd2 +1);

countn2 = zeros(unitHidd2,unitHidd1 +1);
countp2 = zeros(unitHidd2,unitHidd1 +1);

countn1 =zeros(unitHidd1,unitIN +1);
countp1 = zeros(unitHidd1,unitIN +1);

%% to find the number of devices in saturation
Sat_count_Gp3 = zeros(unitOP, unitHidd2 +1);
Sat_count_Gn3 = zeros(unitOP, unitHidd2 +1);

Sat_count_Gp2 = zeros(unitHidd2,unitHidd1 +1);
Sat_count_Gn2 = zeros(unitHidd2,unitHidd1 +1);

Sat_count_Gp1 = zeros(unitHidd1,unitIN +1);
Sat_count_Gn1 =zeros(unitHidd1,unitIN +1);




train = IN1(1:50000,:);
X1 = horzcat(ones(50000,1),train);
Z = X1';
validation = IN1(50001:60000,:);

valid_digit = digit(50001:60000,:);



Nbits = 10;

delta3 = zeros();
DeltaW3 = zeros(unitOP, unitHidd2 +1);
% DeltaW2 = zeros();
%DeltaW2 = zeros(unitHidd,unitIN +1);
countp3 = 0;
countn3 = 0;
countp2 = 0;
countn2 = 0;
countp1 =0;
countn1 =0;

iter = 1; 
maxiter =10;



% C1  = 1; %% gain of the stochastic bit stream for X 
% C2 = 1; %% gain of the stochastic pulse stream small delta
% DW_neg = 0.002;
% DW_pos = 0.002;
alpha1 = 1e4;
alpha2 = 1e4;
beta1 = 1e4;


Rst_L3 = zeros(unitOP, unitHidd2 +1);
Rst_L2 = zeros(unitHidd2,unitHidd1 +1);
Rst_L1 = zeros(unitHidd1,unitIN +1);
ResFreq = 100;

% p1 = -4.7266e-8;
% p2 = 3.1466e-6;
% p3 = 2.717e-5;
% pulse = [0:1:30];
% Gf = p1.*(pulse.^2) + p2.*pulse + p3;
Act_Gmax = 8.21e-5;
Act_Gmin = 4.521e-5;
k = 26; %% no of levels kept same as in the weight bound case
DW_pos = (Act_Gmax - Act_Gmin)./(k-1);
DW_pos = DW_pos;

Gmin = DW_pos.* round(Act_Gmin./DW_pos);
Gmax = DW_pos.* round(Act_Gmax./DW_pos);

allowable_G = Gmin:DW_pos:Gmax;
DW_neg = DW_pos;

SD = 1.5927*DW_neg;
Rst_var = 0.001*DW_neg;

pos_avg_th3 = zeros(1,60000);
pos_avg_th2= pos_avg_th3;
pos_avg_th1= pos_avg_th3;
neg_avg_th3 = pos_avg_th3;
neg_avg_th2 = pos_avg_th2;
neg_avg_th1 = pos_avg_th1;

tic
while (iter<= maxiter)

        ctGp3 = 0;
        ctGn3 = 0;
        
        ctGp2 = 0;
        ctGn2 = 0;
        
        ctGp1 = 0;
        ctGn1 = 0;
        
        %%activation functions 
    for i = 1:length(train) 
        in1 = Z(:,i);
        sign_in1 = sign(in1);
        
        theta1_check = theta1.* sign_in1';
        for k = 1: length(in1)
            in1stream(k,:) = rand(1, Nbits) < abs(in1(k));
        end
        
        In1(:,i)= sum(in1stream')'./Nbits;
        
        z1(:,i) = theta1_check *  In1(:,i);

        output1(:,i) = sigmoid(alpha1.*z1(:,i));

        X2(:,i) = [1 output1(:,i)']';

        in2 = X2(:,i);
        for k = 1: length(in2)
           in2stream(k,:) = rand(1,Nbits) < (in2(k)); 
        end
        

        In2(:,i) = sum(in2stream')'./Nbits;
        z2(:,i) = theta2* In2(:,i);
     
        output2(:,i) = sigmoid(alpha2.*z2(:,i));
        X3(:,i) = [1 output2(:,i)']';
      
        in3 = X3(:,i);
      
        for k = 1: length(in3)
           in3stream(k,:) = rand(1,Nbits) < (in3(k)); 
        end
  
        In3(:,i) = sum(in3stream')'./Nbits;
       
        z3(:,i) = theta3* In3(:,i);
        
        zi = z3(:,i);
        
        for j = 1: unitOP
            output(j,i) = exp(beta1.*zi(j))./sum(exp(beta1.*zi));  %%softmax for the output layer
        end
      
        
        MC(:,i) = -1 * sum(target(:,digit(i)+1).* log(output(:,i)));

  %%back propagation  
  delta3 = (output(:,i)- target(:,digit(i)+1));
%   delta3soft = (outputsoft(:,i)- target(:,digit(i)+1));
 %% to incorporate the learning rate 


if (iter == 1)                                                                                                                         
    C1 = 1; %%
    C2 = 1; %%
  elseif(iter >1 && iter <=2)
    C1 = 0.6;%
    C2 = 0.6;%
  elseif(iter ==3)
    C1 = 0.4;%
    C2 = 0.4;%  
  elseif(iter >3 && iter <=4)
    C1 = 0.3;%
    C2 = 0.3;%
   elseif(iter >4 && iter<=5)
    C1 = 0.2;%
    C2 = 0.2;%
 elseif(iter >5 && iter <=6)
    C1 = 0.15;%
    C2 = 0.15;%
elseif(iter >6 && iter <8)
    C1 = 0.1;%
    C2 = 0.1;%
elseif(iter >=8 && iter <9)
    C1 = 0.05;%
    C2 = 0.05;%
elseif(iter >9 && iter <=10)
    C1 = 0.01;%
    C2 = 0.01;%
  end 


   X3_out = X3(:,i).* C1;
%    X3_check = X3(:,i);
  for j = 1: length(X3_out)
   X3_outstream(j,:) = rand(1,Nbits) < (abs(X3_out(j))); 
  end 
  
  signX3 = sign(X3_out);
  
   for j = 1: unitOP
   delta3_bp(j,:) = rand(1,Nbits) < (abs(delta3(j))); 
   end 
  
  sigd3 = sign(delta3);
  delta3_in = abs(delta3).* C2;
  
  for op=1:unitOP
        delta3stream(op,:) = rand(1,Nbits)<(delta3_in(op));
  end
  


     
  for j = 1: unitOP
   DeltaW3and = delta3stream(j,:).* X3_outstream;
   DeltaW3(j,:) = sum(DeltaW3and'); %% have the total number of coinicidences
  end 
  DeltaW3 = DeltaW3.* sigd3 .*-1;%% negative sign of eta
  
%%%till here %%%%%%
  
  
%    DeltaW3 = DeltaW3.* (1./Nbits).* sigd3 .* signX3';
  % Range_delta3before (:,:,iter)= DeltaW3;
   I3_neg = find(DeltaW3 <0);
   I3_pos = find(DeltaW3 >0);
   DeltaW3(I3_neg) = DW_neg.* ((DeltaW3(I3_neg)));
   DeltaW3(I3_pos) = DW_pos.* (DeltaW3(I3_pos));
   L3p = find(DeltaW3>0);

   
   Gp3(I3_pos) = Gp3(I3_pos) + DeltaW3(I3_pos);
   Gn3(I3_neg) = Gn3(I3_neg) + (abs(DeltaW3(I3_neg)));
   
   

Gp3(I3_pos) = Gp3(I3_pos) + SD.*randn(size(I3_pos));
Gn3(I3_neg) = Gn3(I3_neg) + SD.*randn(size(I3_neg));
   
      Gp3(Gp3>Gmax) = Gmax;
      Gn3(Gn3>Gmax) = Gmax;
   
     Gp3(Gp3< Gmin) = Gmin;
     Gn3(Gn3 <Gmin) = Gmin;
     
  

   
 
            
        if (mod(i,ResFreq)==0)
            Sat_count_Gp3 = zeros(unitOP, unitHidd2 +1);
            Sat_count_Gn3 = zeros(unitOP, unitHidd2 +1);
            diff3 = Gp3 - Gn3;
            inp3 = find(diff3>=0);
            inn3 = find(diff3<0);
            Gp3(inp3)= Gmin.* ones(size(inp3)) + Gp3(inp3) - Gn3(inp3) + Rst_var.*randn(size(inp3));
            Gn3(inp3) = Gmin + Rst_var.*randn(size(inp3));
            
            Rst_L3(inp3) = Rst_L3(inp3) + ones(size(inp3));
            
            Gn3(inn3)= Gmin.* ones(size(inn3)) + Gn3(inn3) - Gp3(inn3)+ Rst_var.*randn(size(inn3));
            Gp3(inn3) = Gmin+ Rst_var.*randn(size(inn3));
            Gp3(Gp3>=Gmax) = Gmax;
            Gn3(Gn3>=Gmax) = Gmax;           
            Gp3(Gp3<=Gmin) = Gmin;
            Gn3(Gn3<=Gmin) = Gmin;
            Rst_L3(inn3) = Rst_L3(inn3) + ones(size(inn3));
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       if((Gp3(Gp3 >= Gmax)))
%           countp3 = countp3 + 1;
% %           if (Gn3(Gn3 >=0.9))
%             in_Gp3 = find(Gp3 >= Gmax);
%             Gp3(in_Gp3) = Gmax;
%             Gp3(in_Gp3)= Gmin.* ones(size(in_Gp3)) + Gp3(in_Gp3) - Gn3(in_Gp3) ;
%             Gn3(in_Gp3) = Gmin;
%             Rst_L3(in_Gp3) = Rst_L3(in_Gp3) + ones(size(in_Gp3));
% 
%       end
%    
% 
%       if ((Gn3(Gn3 >= Gmax)))
%           countn3 = countn3 +1;
% %           if (Gp3(Gp3 >=0.9))
%             
%             in_Gn3 = find(Gn3 >= Gmax);
%             Gn3(in_Gn3) = Gmax;
%             Gn3(in_Gn3)= Gmin.* ones(size(in_Gn3)) + Gn3(in_Gn3) - Gp3(in_Gn3);
%             Gp3(in_Gn3) = Gmin;
%             Rst_L3(in_Gn3) = Rst_L3(in_Gn3) + ones(size(in_Gn3));
%  
%       end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
  
  
  %Range_delta3after(:,:,iter)= DeltaW3; 
   %%introduce stochasticity here 
  smldel3= sum(delta3_bp')'./Nbits; %% equivalent to a forward pass
  theta3_check = theta3.* sigd3;       
  del2 = (theta3_check' * smldel3).*X3(:,i).* (1- X3(:,i)).*alpha2; %%these constants can be included along with the translators
  delta2_new = del2(2:end);
	for j = 1:unitHidd2  
        delta2_bp (j,:) = rand(1,Nbits)<(abs(delta2_new(j)));   %% for back propagation 
	end 
  
  
  
  check = delta2_new.* C2;
  %% probabilistic weight update here 
  %%bit stream for delta2_new
%   for j = 1:unitHidd2  
%         delta2stream1 (j,:) = rand(1,Nbits)<(abs(delta2_new(j)));
%   end 
  
    for j = 1:unitHidd2  
        delta2stream (j,:) = rand(1,Nbits)<(abs(check(j)));
    end 
    X2_out = X2(:,i);
    check1 = X2_out.* C1;
  for j = 1: length(X2_out)
    X2_outstream(j,:) = rand(1,Nbits) < (abs(check1(j))); 
  end  
  
%     for j = 1: length(X2_out)
%     X2_outstream1(j,:) = rand(1,Nbits) < (abs(X2_out(j))); 
%   end  
    sigd2 = sign(delta2_new);
%     signX2 = sign(X2_out);
    

for j = 1: unitHidd2
   DeltaW2and = delta2stream(j,:) .* X2_outstream; 
   DeltaW2(j,:) = sum(DeltaW2and'); 
end 
%    DeltaW2 = DeltaW2.* (1./Nbits).* sigd2 .* signX2';
   DeltaW2 = DeltaW2.* sigd2 .*-1; %% negative sign of eta
   
%     Range_delta2before (:,:,iter)= DeltaW2;
   I2_neg = find(DeltaW2 <0);
   I2_pos = find(DeltaW2 >0);
   DeltaW2(I2_neg) = DW_neg.*((DeltaW2(I2_neg)));
   DeltaW2(I2_pos) = DW_pos.*(DeltaW2(I2_pos));
   

   
   
    
   Gp2(I2_pos) = Gp2(I2_pos) + DeltaW2(I2_pos);
   Gn2(I2_neg) = Gn2(I2_neg) + (abs(DeltaW2(I2_neg)));
   
 
   
%  Gp2(I2_pos) = DW_pos.*round(Gp2(I2_pos)./DW_pos) + SD.*randn(size(I2_pos));
%  Gn2(I2_neg) = DW_neg.*round(Gn2(I2_neg)./DW_neg)+ SD.*randn(size(I2_neg));
     Gp2(I2_pos) =  Gp2(I2_pos) + SD.*randn(size(I2_pos));
      Gn2(I2_neg) =  Gn2(I2_neg) + SD.*randn(size(I2_neg));

  
    Gp2(Gp2>Gmax) = Gmax;
    Gn2(Gn2>Gmax) = Gmax;
  
   Gp2(Gp2< Gmin) = Gmin;
   Gn2(Gn2 <Gmin) = Gmin;
  
  


   
          if (mod(i,ResFreq)==0)
               
            Sat_count_Gp2 = zeros(unitHidd2,unitHidd1 +1);
            Sat_count_Gn2 = zeros(unitHidd2,unitHidd1 +1);
            diff2 = Gp2 - Gn2;
            inp2 = find(diff2>=0);
            inn2 = find(diff2<0);
            Gp2(inp2 )= Gmin.* ones(size(inp2)) + Gp2(inp2) - Gn2(inp2)+ Rst_var.*randn(size(inp2)) ;
            Gn2(inp2) = Gmin+ Rst_var.*randn(size(inp2));
           
            Rst_L2(inp2) = Rst_L2(inp2) + ones(size(inp2));
            
            Gn2(inn2)= Gmin.* ones(size(inn2)) + Gn2(inn2) - Gp2(inn2)+ Rst_var.*randn(size(inn2));
            Gp2(inn2) = Gmin + Rst_var.*randn(size(inn2));
            Gp2(Gp2>=Gmax) = Gmax;
            Gn2(Gn2>=Gmax) = Gmax;           
            Gp2(Gp2<=Gmin) = Gmin;
            Gn2(Gn2<=Gmin) = Gmin;
            
            Rst_L2(inn2) = Rst_L2(inn2) + ones(size(inn2));
          end
   
   
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       if((Gp2(Gp2 >= Gmax)))
%           countp2 = countp2 + 1;
% %           if (Gn2(Gn2 >=0.9))
%             
%             in_Gp2 = find(Gp2 >= Gmax);
%             Gp2(in_Gp2) = Gmax;
%             Gp2(in_Gp2)= Gmin.* ones(size(in_Gp2)) + Gp2(in_Gp2) - Gn2(in_Gp2);
%             Gn2(in_Gp2) = Gmin;
%             Rst_L2(in_Gp2) = Rst_L2(in_Gp2) + ones(size(in_Gp2));
% 
%       end
%    
%       if ((Gn2(Gn2 >= Gmax)))
%           countn2 = countn2 +1;
% %           if (Gp2(Gp2 >=0.9))
%             
%             in_Gn2 = find(Gn2 >= Gmax);
%             Gn2(in_Gn2) = Gmax;
%             Gn2(in_Gn2)= Gmin.* ones(size(in_Gn2)) + Gn2(in_Gn2) - Gp2(in_Gn2);
%             Gp2(in_Gn2) = Gmin;
%             Rst_L2(in_Gn2) = Rst_L2(in_Gn2) + ones(size(in_Gn2));
% 
%       end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  





 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  
  
   %%% for layer 2 
   back1 = X2(:,i).* (1- X2(:,i));
  smldel2= sum(delta2_bp')'./Nbits;
  theta2_check = theta2.* sigd2;       
  del1 = (theta2_check' * smldel2).* back1 .*alpha1; %%these constants can be included along with the translators
  delta1_new = del1(2:end);
  check_delta1 = delta1_new.* C1;
  %% probabilistic weight update here 
%   DeltaW1 = delta1_new * Z(:,i)';
    
 %%bit stream for delta1_new
%   for j = 1:unitHidd1  
%         delta1stream1 (j,:) = rand(1,Nbits)<(abs(delta1_new(j)));
%   end 
    for j = 1:unitHidd1  
        delta1stream (j,:) = rand(1,Nbits)<(abs(check_delta1(j)));
  end 
  
    Z_in = Z(:,i);
    check_Zin = Z_in .* C2;
  for j = 1: length(Z_in)
    Z_instream(j,:) = rand(1,Nbits) < (abs(check_Zin(j))); 
  end   
    
%     for j = 1: length(Z_in)
%     Z_instream1(j,:) = rand(1,Nbits) < (abs(Z_in(j))); 
%   end  
    sigd1 = sign(delta1_new);
    signZ = sign(Z_in);


for j = 1: unitHidd1
   DeltaW1and = delta1stream(j,:) .* Z_instream; 
   DeltaW1(j,:) = sum(DeltaW1and'); 
end 
   DeltaW1 = DeltaW1.* sigd1 .* signZ'.*-1;
 
%   Range_delta1before (:,:,iter)= DeltaW1;
   I1_neg = find(DeltaW1 <0);
   I1_pos = find(DeltaW1 >0);
   DeltaW1(I1_neg) = DW_neg.*((DeltaW1(I1_neg)));
   DeltaW1(I1_pos) = DW_pos.* (DeltaW1(I1_pos));
   

   
   
    Gp1(I1_pos) = Gp1(I1_pos) + DeltaW1(I1_pos);
   Gn1(I1_neg) = Gn1(I1_neg) + abs(DeltaW1(I1_neg));
   
 
   
   
%     Gp1(I1_pos) = DW_pos.* round(Gp1(I1_pos)./DW_pos)+ SD.*randn(size(I1_pos));
%     Gn1(I1_neg) = DW_neg.*round(Gn1(I1_neg)./DW_neg)+ SD.*randn(size(I1_neg));
 Gp1(I1_pos) =  Gp1(I1_pos) + SD.*randn(size(I1_pos));
  Gn1(I1_neg) =  Gn1(I1_neg) + SD.*randn(size(I1_neg));
   
   Gp1(Gp1>Gmax) = Gmax;
   Gn1(Gn1>Gmax) = Gmax;
   
   Gp1(Gp1< Gmin) = Gmin;
   Gn1(Gn1<Gmin) = Gmin;
   
  
 
   
   if (mod(i,ResFreq)==0)
               
            Sat_count_Gp1 = zeros(unitHidd1,unitIN +1);
            Sat_count_Gn1 =zeros(unitHidd1,unitIN +1);
            diff1 = Gp1 - Gn1;
            inp1 = find(diff1>=0);
            inn1 = find(diff1<0);
            Gp1(inp1)= Gmin.* ones(size(inp1)) + Gp1(inp1) - Gn1(inp1) + Rst_var.*randn(size(inp1)) ;
            Gn1(inp1) = Gmin + Rst_var.*randn(size(inp1));
           
            Rst_L1(inp1) = Rst_L1(inp1) + ones(size(inp1));
            
            Gn1(inn1)= Gmin.* ones(size(inn1)) + Gn1(inn1) - Gp1(inn1)+ Rst_var.*randn(size(inn1));
            Gp1(inn1) = Gmin+ Rst_var.*randn(size(inn1));
            Gp1(Gp1>=Gmax) = Gmax;
            Gn1(Gn1>=Gmax) = Gmax;           
            Gp1(Gp1<=Gmin) = Gmin;
            Gn1(Gn1<=Gmin) = Gmin;
            
            Rst_L1(inn1) = Rst_L1(inn1) + ones(size(inn1));
            
    end
   
   
   
%  %%%%%%%%%%%%%%%%%%%%%%%%%%Cond reset%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%       if((Gp1(Gp1 >= Gmax)))
%           countp1 = countp1 + 1;
% %           if (Gn1(Gn1 >=0.9))
%             in_Gp1 = find(Gp1 >= Gmax);
%             Gp1(in_Gp1) = Gmax;
%             Gp1(in_Gp1)= Gmin.* ones(size(in_Gp1)) + Gp1(in_Gp1) - Gn1(in_Gp1);
%             Gn1(in_Gp1) = Gmin;
%             Rst_L1(in_Gp1) = Rst_L1(in_Gp1) + ones(size(in_Gp1));
% 
%       end
%    
%    
%       if ((Gn1(Gn1 >= Gmax)))
%           countn1 = countn1 +1;
% %           if (Gp1(Gp1 >=0.9))
%             in_Gn1 = find(Gn1 >= Gmax);
%             Gn1(in_Gn1) = Gmax;
%             Gn1(in_Gn1)= Gmin.* ones(size(in_Gn1)) + Gn1(in_Gn1) - Gp1(in_Gn1);
%             Gp1(in_Gn1) = Gmin;
%             Rst_L1(in_Gn1) = Rst_L1(in_Gn1) + ones(size(in_Gn1));
% 
%       end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   


   theta1 =  Gp1 - Gn1;
   theta2 =  Gp2 - Gn2;
   theta3 =  Gp3 - Gn3;
   
   avg_th3(:,i) = sum(theta3(:))./1290;
   avg_th2(:,i) = sum(theta2(:))./32896;
   avg_th1(:,i) = sum(theta1(:))./200960;
   

%    
       theta1_max (i,iter) = max(theta1(:));
       theta2_max (i,iter) = max(theta2(:));
       theta3_max (i,iter) = max(theta3(:));
       
       theta1_min (i,iter) = min(theta1(:));
       theta2_min (i,iter) = min(theta2(:));
       theta3_min (i,iter) = min(theta3(:));
       
   
        
%        count = count + 1;
    end
    
   
   
   J(iter) = sum(MC)./length(train)
   %% Accuracy check 
   % validation check (5000 images in the training set)
   valid_set = horzcat(ones(length(validation),1),validation);%IN_test
   Z_valid = valid_set';
   valid_samples =length(validation);
   
    alpha1 = 1e4;
    alpha2 = 1e4;
    beta1 = 1e4; 
   
   Nbits = 10; 
   
      for ii = 1:length(validation) 
        
        in1_valid = Z_valid(:,ii);
        sign_in1 = sign(in1_valid);
        theta1_check = theta1.* sign_in1';
        for k = 1: length(in1_valid)
            in1stream(k,:) = rand(1, Nbits) < abs(in1_valid(k));
        end
        
        In1(:,ii)= sum(in1stream')'./Nbits;
        z1(:,ii) = theta1_check *  In1(:,ii);
        output1(:,ii) = sigmoid(alpha1.*z1(:,ii));
        X2_valid(:,ii) = [1 output1(:,ii)']';
        
        in2 = X2_valid(:,ii);
        for k = 1: length(in2)
           in2stream(k,:) = rand(1,Nbits) < (in2(k)); 
        end
        In2(:,ii) = sum(in2stream')'./Nbits;
        
        
        z2(:,ii) = theta2* In2(:,ii);
        output2(:,ii) = sigmoid(alpha2.*z2(:,ii));
        X3_valid(:,ii) = [1 output2(:,ii)']';
        in3 = X3_valid(:,ii);
        for k = 1: length(in3)
           in3stream(k,:) = rand(1,Nbits) < (in3(k)); 
        end
        In3(:,ii) = sum(in3stream')'./Nbits;
        
        
        z3(:,ii) = theta3* In3(:,ii);
        zi = z3(:,ii);
        for j = 1: unitOP
            output_valid(j,ii) = exp(beta1*zi(j))./sum(beta1.*exp(zi));  %%softmax for the output layer
        end
        
      end
     
 [M I] = max(output_valid);
 final_valid = I-1;
 check_valid = final_valid'- valid_digit(1:length(validation));
 Acc_valid(iter) = length(find(check_valid==0)).* 100./length(validation)
 
 
   
 
    %%% to determine the the test accuracy after each epoch 
    
   load ./../test1.mat
   load ./../test_digit.mat 
   X1_test = horzcat(ones(10000,1),test);
   Z_test = X1_test';
   test_samples =10000;
   
    alpha1 = 1e4;
    alpha2 = 1e4;
    beta1 = 1e4; 
    

   
   Nbits = 10; 
   
      for ii = 1:test_samples 
        
        in1_test = Z_test(:,ii);
        sign_in1 = sign(in1_test);
        theta1_check = theta1.* sign_in1';
        for k = 1: length(in1_test)
            in1stream(k,:) = rand(1, Nbits) < abs(in1_test(k));
        end
        
        In1(:,ii)= sum(in1stream')'./Nbits;
        z1(:,ii) = theta1_check *  In1(:,ii);
        output1(:,ii) = sigmoid(alpha1.*z1(:,ii));
        X2_test(:,ii) = [1 output1(:,ii)']';
        
        in2 = X2_test(:,ii);
        for k = 1: length(in2)
           in2stream(k,:) = rand(1,Nbits) < (in2(k)); 
        end
        In2(:,ii) = sum(in2stream')'./Nbits;
        
        
        z2(:,ii) = theta2* In2(:,ii);
        output2(:,ii) = sigmoid(alpha2.*z2(:,ii));
        X3_test(:,ii) = [1 output2(:,ii)']';
        in3 = X3_test(:,ii);
        for k = 1: length(in3)
           in3stream(k,:) = rand(1,Nbits) < (in3(k)); 
        end
        In3(:,ii) = sum(in3stream')'./Nbits;
        
        
        z3(:,ii) = theta3* In3(:,ii);
        zi = z3(:,ii);
        for j = 1: unitOP
            output_test(j,ii) = exp(beta1*zi(j))./sum(beta1.*exp(zi));  %%softmax for the output layer
        end
        
      end
     
 [M I] = max(output_test);
 final_test = I-1;
 check_test = final_test'- test_digit(1:test_samples);
 Acc_test(iter) = length(find(check_test==0)).* 100./test_samples
 

  
  X1_train = horzcat(ones(60000,1),IN1);
  Z_train = X1_train';
 alpha = 1e4;
 beta = 1e4; 
 train_samples = 60000;
 
  Nbits = 10; 
   
      for kk = 1:train_samples 
        
        in1_train = Z_train(:,kk);
        sign_in1 = sign(in1_train);
        theta1_check = theta1.* sign_in1';
        for k = 1: length(in1_train)
            in1stream(k,:) = rand(1, Nbits) < abs(in1_train(k));
        end
        
        In1(:,kk)= sum(in1stream')'./Nbits;
        z1(:,kk) = theta1_check *  In1(:,kk);
        output1(:,kk) = sigmoid(alpha1.*z1(:,kk));
        X2_train(:,kk) = [1 output1(:,kk)']';
        
        in2 = X2_train(:,kk);
        for k = 1: length(in2)
           in2stream(k,:) = rand(1,Nbits) < (in2(k)); 
        end
        In2(:,kk) = sum(in2stream')'./Nbits;
        
        
        z2(:,kk) = theta2* In2(:,kk);
        output2(:,kk) = sigmoid(alpha2.*z2(:,kk));
        X3_train(:,kk) = [1 output2(:,kk)']';
        in3 = X3_train(:,kk);
        for k = 1: length(in3)
           in3stream(k,:) = rand(1,Nbits) < (in3(k)); 
        end
        In3(:,kk) = sum(in3stream')'./Nbits;
        
        
        z3(:,kk) = theta3* In3(:,kk);
        zi = z3(:,kk);
        for j = 1: unitOP
            output_train(j,kk) = exp(beta1*zi(j))./sum(beta1.*exp(zi));  %%softmax for the output layer
        end
        
      end
     
 [M Itrain] = max(output_train);
final_train = Itrain-1;
 check_train = final_train'- digit(1:numOfSamples);
 Acc_train (iter) = length(find(check_train==0)).* 100./train_samples
   iter = iter +1;
   
end 

max_theta1 = max(theta1_max(:))
max_theta2 = max(theta2_max(:))
max_theta3 = max(theta3_max(:))

min_theta1 = min(theta1_min(:))
min_theta2 = min(theta2_min(:))
min_theta3 = min(theta3_min(:))


toc
set(0,'defaulttextinterpreter','tex');
set(groot, 'defaultAxesTickLabelInterpreter','tex'); set(groot, 'defaultLegendInterpreter','tex');

figure (2)
semilogy(J,'-.r*','linewidth',2); grid on 
set(gca,'Fontsize',16);
title('Cost function');
xlabel(' Epochs');
ylabel('Average Crossentropy Error');
hold off
% saveas(gcf,'Cost100_x_sig_PCMO_seed52_final.png');

figure (3) 
plot( Acc_train,'r-o','MarkerFaceColor','none','linewidth',2); grid on 
hold on 
set(gca,'Fontsize',17,'FontWeight','bold','linewidth',1);
xlabel('Epochs');
ylabel('Accuracy (%)');
% title('Accuracy');

% figure (3) 
plot( Acc_test,'black-o','MarkerFaceColor','none','linewidth',2); grid on
set(gca,'Fontsize',17,'FontWeight','bold','linewidth',1);
xlabel('Epochs');
ylabel('Accuracy (%)');
% title('Accuracy');

legend('Training Set','Test Set','location','best');
% saveas(gcf,'train_100x_sig_PCMO_seed52_final.png');
% save('Cost_100x_seed52_final.mat','J')
%to save the output files 
% mkdir With_Qtzn
save('Acc_train_onoff1.8_sigma_PCMO_reset100_x_resetvar_0.001_seed52_final.mat','Acc_train');
save('Acc_test_onoff1.8_sigma_PCMO_reset100_x_resetvar_0.001_seed52_final.mat','Acc_test');
save('Acc_valid_onoff1.8_sigma_PCMO_reset100_x_resetvar_0.001_seed52_final.mat','Acc_valid');
save('theta1_stoch_trained_onoff1.8_sigma_PCMO_reset100_x_resetvar_0.001_seed52_final.mat','theta1');
save('theta2_stoch_trained_onoff1.8_sigma_PCMO_reset100_x_resetvar_0.001_seed52_final.mat','theta2');
save('theta3_stoch_trained_onoff1.8_sigma_1pt5B_reset100_x_resetvar_0.001_seed52_final.mat','theta3');

