function berfin_kavsut_21602459_hw2(question)
clc
close all

switch question
    
    case '1'
	disp('Q1')
    
    %read dataset  
    trainims = h5read('assign2_data1.h5','/trainims');
    trainlbls = h5read('assign2_data1.h5','/trainlbls');    
    testims = h5read('assign2_data1.h5','/testims');
    testlbls = h5read('assign2_data1.h5','/testlbls');
    
    trainlbls(trainlbls == 0) = -1;
    testlbls(testlbls == 0) = -1;
    
    %data type conversion 
    train_images = im2double(trainims);   
    test_images = im2double(testims);   
    
    %size of images, training image number and test image number 
    [m,n,train_no] = size(train_images);
    [~,~,test_no] = size(test_images);
    
    %training set and test set 
    X_train = reshape(train_images,m*n,train_no);
    X_test = reshape(test_images,m*n,test_no);
    y_train = trainlbls;
    y_test = testlbls; 
    
    %parameters     
    lr = 0.15; % learning rate in range of [0.1,0.5]
    batch_size = 100; %mini batch size 
    epoch_no = 500; 
    
    class_no = 2; %output neuron number 
    
    disp('Part A')
    N = 10; %hidden layer neuron number 
    history = neural_net_Q1(X_train, y_train, X_test, y_test, batch_size,lr,epoch_no,class_no,N);
 
    %display error metrics for Part A 
    figure;set(gcf, 'WindowState', 'maximized');
    subplot(2,1,1); plot(1:epoch_no, history.train_MSE);
    hold on; plot(1:epoch_no,history.test_MSE);
    legend('Training Set','Test Set')
    title('Mean Squared Error');
    
    subplot(2,1,2); plot(1:epoch_no, history.train_MCE);
    hold on; plot(1:epoch_no,history.test_MCE);
    legend('Training Set','Test Set')
    title('Classification Error')    
    %saveas(gcf,'Q1_Part_A.png');
    
    disp('Part B')
    disp('In the report.')
    
    disp('Part C')
    N_opt = 10; 
    N_low = 2;
    N_high = 50;    
    history_low = neural_net_Q1(X_train, y_train, X_test, y_test, batch_size,lr,epoch_no,class_no,N_low);   
    history_opt = neural_net_Q1(X_train, y_train, X_test, y_test, batch_size,lr,epoch_no,class_no,N_opt);
    history_high = neural_net_Q1(X_train, y_train, X_test, y_test, batch_size,lr,epoch_no,class_no,N_high);

    figure;set(gcf, 'WindowState', 'maximized');
    subplot(2,2,1); plot(1:epoch_no, history_low.train_MSE)
    hold on; plot(1:epoch_no, history_opt.train_MSE)
    hold on; plot(1:epoch_no, history_high.train_MSE)
    legend('N_{low}','N^*','N_{high}')
    title('Mean Squared Error of Training Set')
    
    subplot(2,2,2); plot(1:epoch_no, history_low.train_MCE)
    hold on; plot(1:epoch_no, history_opt.train_MCE)
    hold on; plot(1:epoch_no, history_high.train_MCE)
    legend('N_{low}','N^*','N_{high}')
    title('Classification Error of Training Set')
    
    subplot(2,2,3); plot(1:epoch_no, history_low.test_MSE)
    hold on; plot(1:epoch_no, history_opt.test_MSE)
    hold on; plot(1:epoch_no, history_high.test_MSE)
    legend('N_{low}','N^*','N_{high}')
    title('Mean Squared Error of Test Set')
    
    subplot(2,2,4); plot(1:epoch_no, history_low.test_MCE)
    hold on; plot(1:epoch_no, history_opt.test_MCE)
    hold on; plot(1:epoch_no, history_high.test_MCE)
    legend('N_{low}','N^*','N_{high}')
    title('Classification Error of Test Set')
    %saveas(gcf,'Q1_Part_C.png');
    
    disp('Part D')
    
    lr = 0.15;
    alpha = 0; %without momentum 
    N1 = 10;
    N2 = 10;
    history = neural_net2_Q1(X_train, y_train, X_test, y_test, batch_size,...
    lr,alpha,epoch_no,class_no,N1,N2);
    
    %error metrics for part d 
    figure;set(gcf, 'WindowState', 'maximized');
    subplot(2,1,1); plot(1:epoch_no, history.train_MSE);
    hold on; plot(1:epoch_no,history.test_MSE);
    legend('Training Set','Test Set')
    title('Mean Squared Error (without momentum)');
    
    subplot(2,1,2); plot(1:epoch_no, history.train_MCE);
    hold on; plot(1:epoch_no,history.test_MCE);
    legend('Training Set','Test Set')
    title('Classification Error (without momentum)')
    %saveas(gcf,'Q1_Part_D.png');
    
    disp('Part E')
    
    lr = 0.15;
    alpha = 0.5;
    N1 = 10;
    N2 = 10;
    history = neural_net2_Q1(X_train, y_train, X_test, y_test, batch_size,...
    lr,alpha,epoch_no,class_no,N1,N2);
    
    %error metrics for part e 
    figure;set(gcf, 'WindowState', 'maximized');
    subplot(2,1,1); plot(1:epoch_no, history.train_MSE);
    hold on; plot(1:epoch_no,history.test_MSE);
    legend('Training Set','Test Set')
    title('Mean Squared Error  (with momentum)');
    
    subplot(2,1,2); plot(1:epoch_no, history.train_MCE);
    hold on; plot(1:epoch_no,history.test_MCE);
    legend('Training Set','Test Set')
    title('Classification Error (with momentum)')
    %saveas(gcf,'Q1_Part_E.png');
    
    case '2'
    disp('Q2')
    
    testd = h5read('assign2_data2.h5','/testd');
    testx = h5read('assign2_data2.h5','/testx');    
    traind = h5read('assign2_data2.h5','/traind');
    trainx = h5read('assign2_data2.h5','/trainx');
    vald = h5read('assign2_data2.h5','/vald');
    valx = h5read('assign2_data2.h5','/valx');
    
    disp('Part A')
    lr = 0.15; 
    alpha = 0.85;
    epoch_no = 50; 
    batch_size = 200; 
    
    %(D,P) = (32,256)
    D = 32; 
    P = 256; 
    [history,~] = neural_net_Q2(trainx,traind,valx,vald,testx,testd,epoch_no,batch_size,lr,alpha,D,P);
    
    figure;set(gcf, 'WindowState', 'maximized');
    subplot(2,1,1); plot(1:history.epoch,history.train_CE);
    hold on; plot(1:history.epoch,history.val_CE);
    title('Cross Entropy Error (D,P) = (32,256)')
    legend('Training Set','Validation Set')
    
    subplot(2,1,2); plot(1:history.epoch,history.train_MCE);
    title('Classification Error (D,P) = (32,256)')
    hold on; plot(1:history.epoch,history.val_MCE);    
    legend('Training Set','Validation Set')
    %saveas(gcf,'Q2_(32,256).png')
    
    %(D,P) = (16,128)
    D = 16; 
    P = 128; 
    [history,weights] = neural_net_Q2(trainx,traind,valx,vald,testx,testd,epoch_no,batch_size,lr,alpha,D,P);
    
    figure;set(gcf, 'WindowState', 'maximized');
    subplot(2,1,1); plot(1:history.epoch,history.train_CE);    
    title('Cross Entropy Error (D,P) = (16,128)')
    hold on; plot(1:history.epoch,history.val_CE);
    legend('Training Set','Validation Set')
    
    subplot(2,1,2); plot(1:history.epoch,history.train_MCE); 
    title('Classification Error (D,P) = (16,128)')
    hold on; plot(1:history.epoch,history.val_MCE);    
    legend('Training Set','Validation Set')
    %saveas(gcf,'Q2_(16,128).png')

    %(D,P) = (8,64)
    D = 8; 
    P = 64; 
    [history,~] = neural_net_Q2(trainx,traind,valx,vald,testx,testd,epoch_no,batch_size,lr,alpha,D,P);
    
    figure;set(gcf, 'WindowState', 'maximized');
    subplot(2,1,1); plot(1:history.epoch,history.train_CE);    
    title('Cross Entropy Error (D,P) = (8,64)')
    hold on; plot(1:history.epoch,history.val_CE);    
    legend('Training Set','Validation Set')
    
    subplot(2,1,2); plot(1:history.epoch,history.train_MCE);      
    title('Classification Error (D,P) = (8,64)')
    hold on; plot(1:history.epoch,history.val_MCE);    
    legend('Training Set','Validation Set')
    %saveas(gcf,'Q2_(8,64).png')

    disp('Part B')
    
    WE = weights.WE;
    W1 = weights.W1;
    W2 = weights.W2; 
    
    test_no = size(testd,1);
    [X1_test,X2_test,X3_test,D_out_test] = prepare_words(testx,testd);
     
    %Select 5 3-words sequences and their desired outputs  
    indices = 7500*[1:5];
    X1 = X1_test(:,indices);
    X2 = X2_test(:,indices);
    X3 = X3_test(:,indices);
    D_out = testd(indices);
    
    %TEST SET FORWARD PROPAGATION 
    E1 = WE*X1;
    E2 = WE*X2;
    E3 = WE*X3;     
    E = [E1;E2;E3;ones(1,5)]; %bias term  
    V1 = W1*E; 
    [Y1,~] = sigmoid(V1,1);
    Y1 = [Y1;ones(1,5)]; %bias term 
    V2 = W2*Y1;
    Y2 = softmax(V2); %the predicted probability for each of the 250 words is Y2 
    
    for i = 1:5
        
        disp(strcat('Triagram #',num2str(i),':'));
        
        disp('Desired Output:');     
        desired_output = testd(i);
        disp(desired_output);
        
        disp('Top 10 Candidates for the Fourth Word:');
        [sorted,index] = sort(Y2(:,i),'descend'); %sort in descending order
        top_10 = index(1:10,:);%show indices with 10 highest probabilities
        disp(top_10);
        
    end 
       
    case '3'
    disp('Q3')
    disp('In the report.')    
    
end

end 

function [X1,X2,X3,D] = prepare_words(x,d)
    %Create word vectors with one hot representation 
    %Inputs 
    %x: three words sequence
    %d: forth word of the sequence 
    word_no = max(d);
    sample_no = size(d,1);
    X1 = zeros(word_no,sample_no);
    X2 = zeros(word_no,sample_no);
    X3 = zeros(word_no,sample_no);
    D = zeros(word_no,sample_no);    
    for i =1:sample_no 
        X1(x(1,i),i) = 1;
        X2(x(2,i),i) = 1;
        X3(x(3,i),i) = 1;
        D(d(i),i) = 1;
    end     
end     
    
function  CE = cross_entropy(y,y_hat)
    %Inputs 
    %y_hat: the estimate of model  
    %y: the desired output of model  
    CE = -sum(y.*log(y_hat));
end 

function o = softmax(x)   
    %Sofmax operation 
    o = exp(x)./sum(exp(x));
end 

function [o,der] = sigmoid(v,lambda)
    %Unipolar sigmoid activation function 
    %Results are on interval of [0,1]
    %Inputs
    %v: inputs, lambda: parameter for sigmoid, T: threshold 
    if(lambda > 0)
        o = 1./(1+exp(-lambda*v));
        der = o.*(1-o);
    else
        o = nan;
        der = nan;
        disp('Lambda should be a positive value!');
    end     
end

 
function history = neural_net_Q1(X_train, y_train, X_test, y_test, batch_size,...
    lr,epoch_no,class_no,hidden_neuron_no)

%Output layer neuron number 
M = class_no; 
%Hidden layer neuron number 
L = hidden_neuron_no;

%Take sizes 
input_size = size(X_train,1);
train_no = size(X_train,2);

%Choose weights and biases initally, small and random to prevent saturation 
std = 0.001;
W_hidden = std*randn(L,input_size); 
bias_hidden = std*randn(L,1);
W_output = std*randn(M,L);
bias_output = std*randn(M,1);

delta_We_output = zeros(M,L+1);    
delta_We_hidden = zeros(L,input_size+1);

%Batch number with batch size 
batch_no = floor(train_no / batch_size); % B corresponds to batch_size in comments 

%Normalize images 
X_train = X_train./max(X_train); 
X_test = X_test./max(X_test); 

%Train set number and test set number 
train_no = size(X_train,2);
test_no = size(X_test,2);

for N = 1:epoch_no

    %Shuffle training images (shuffle indices)
    indices = 1:train_no;
    indices = indices(randperm(train_no));    

    for j=1:batch_no           
          
        %Take images for each batch 
        X_indices = indices( (j-1)*batch_size+1 : j*batch_size );
        X = X_train(:,X_indices);

        %FORWARD PROPAGATION 
        %input matrix with bias terms
        X = [X;1*ones(1,batch_size)]; %size: 1025xB
        %linear activation potential of hidden layer 
        U = [W_hidden bias_hidden]*X; %size: LxB = (Lx1025)*(1025xB)
        %output of hidden layer, hidden signal vector 
        H = tanh(U); %size: LxB 

        %linear activation potential of output layer 
        V = [W_output bias_output]*[H;1*ones(1,batch_size)]; % size: MxB = (M x L+1)*(L+1 x B)
        %output of output layer, output vector 
        Y = tanh(V);  %size: MxB

        %BACK PROPAGATION
        %Desired output 
        classes = y_train(X_indices);
        D = zeros(class_no,batch_size);
        D(:,classes == 1) = ones(class_no,sum(classes == 1)).*[1;-1]; %car neuron = 1, cat neuron = -1 
        D(:,classes == -1) = ones(class_no,sum(classes == -1)).*[-1;1]; %car neuron = -1, cat neuron = 1

        %LOCAL GRADIENTS OF OUTPUT LAYER 
        %gradient descent update of output weight matrix 
        error_output = D-Y; %size: MxB 
        %derivative of y with respect to v 
        der_of_Y_with_V = 1-Y.^2; %size: MxB   
        %local gradients for the output layer     
        delta_output = error_output.*der_of_Y_with_V;%size: MxB

        %LOCAL GRADIENTS OF HIDDEN LAYER 
        %derivative of h with respect to u 
        der_of_H_with_U = 1-H.^2;  %size: LxB        
        %local gradients for the hidden layer 
        error_hidden = W_output'*delta_output; %size: LxB = (LxM)*(MxB)            
        %local gradient for the hidden layer  
        delta_hidden = error_hidden.*der_of_H_with_U; %size: LxB 

        %WEIGHT UPDATES 
        delta_We_output =  lr*delta_output*[H;-1*ones(1,batch_size)]'/batch_size; %size: MxL = (MxB)*(BxL)         
        delta_We_hidden=  lr*delta_hidden*X'/batch_size; %size: MxL = (Mx1)*(1xL) 

        %UPDATE OUTPUT LAYER WEIGHT MATRIX 
        We_output = [W_output bias_output] + delta_We_output; %MxL
        W_output = We_output(:,1:end-1); bias_output = We_output(:,end);

        %UPDATE HIDDEN LAYER WEIGHT MATRIX 
        We_hidden = [W_hidden bias_hidden] + delta_We_hidden; %size: Lx1025        
        W_hidden = We_hidden(:,1:end-1); bias_hidden = We_hidden(:,end);

    end

    %ERROR METRICS  
    
    %TRAIN SET 
    X = [X_train;1*ones(1,train_no)];
    U = [W_hidden bias_hidden]*X; 
    H = tanh(U); 
    V = [W_output bias_output]*[H;1*ones(1,train_no)]; 
    Y = tanh(V);  
    
    %Desired Output
    classes = y_train;
    D = zeros(class_no,train_no);
    D(:,classes == 1) = ones(class_no,sum(classes == 1)).*[1;-1]; %car neuron = 1, cat neuron = -1 
    D(:,classes == -1) = ones(class_no,sum(classes == -1)).*[-1;1]; %car neuron = -1, cat neuron = 1

    %ERROR METRICS FOR ONE EPOCH     
    %Mean Squared Error
    MSE = sum(sum(0.5*(D-Y).^2));
    MSE = MSE/train_no;
    %Mean Classification Error         
    [~,real_classes] = max(D);
    [~,pred_classes] = max(Y);
    MCE = sum(real_classes ~= pred_classes);
    MCE = MCE/train_no * 100;

    %Record error
    history.train_MSE(N) = MSE;
    history.train_MCE(N) = MCE;
    
    %TEST SET     
    X = [X_test;1*ones(1,test_no)]; 
    U = [W_hidden bias_hidden]*X; 
    H = tanh(U);  
    V = [W_output bias_output]*[H;1*ones(1,test_no)]; 
    Y = tanh(V);  
    
    %Desired Output
    classes = y_test;
    D = zeros(class_no,test_no);
    D(:,classes == 1) = ones(class_no,sum(classes == 1)).*[1;-1]; %car neuron = 1, cat neuron = -1 
    D(:,classes == -1) = ones(class_no,sum(classes == -1)).*[-1;1]; %car neuron = -1, cat neuron = 1
             
    %Mean Squared Error
    MSE = sum(sum(0.5*(D-Y).^2))/test_no;
    %Mean Classification Error
    [~,real_classes] = max(D);
    [~,pred_classes] = max(Y);   
    MCE = sum(real_classes ~= pred_classes)/test_no * 100;
        
    %Record error
    history.test_MSE(N) = MSE;
    history.test_MCE(N) = MCE;
    
end 

end 
   
function history = neural_net2_Q1(X_train, y_train, X_test, y_test, batch_size,...
    lr,alpha,epoch_no,class_no,hidden_neuron_no,hidden_neuron_no2)
%This function is addition of one more hidden layer in neural_net_Q1

M = class_no; 
L = hidden_neuron_no2;
K = hidden_neuron_no;

input_size = size(X_train,1);
train_no = size(X_train,2);

std = 0.01;
W_hidden1 = std*randn(K,input_size); 
bias_hidden1 = std*randn(K,1);
W_hidden2 = std*randn(L,K); 
bias_hidden2 = std*randn(L,1);
W_output = std*randn(M,L);
bias_output = std*randn(M,1);

delta_We_output = zeros(M,L+1);    
delta_We_hidden1 = zeros(K,input_size+1);
delta_We_hidden2 = zeros(L,K+1);

batch_no = floor(train_no / batch_size); 

X_test = X_test./max(X_test);
X_train = X_train./max(X_train);

test_no = size(X_test,2);
train_no = size(X_train,2);

for N = 1:epoch_no

    %shuffle training images 
    indices = 1:train_no;
    indices = indices(randperm(train_no));    

    for j=1:batch_no           
          
        X_indices = indices( (j-1)*batch_size+1 : j*batch_size );
        X = X_train(:,X_indices);
    
        %FORWARD PROPAGATION 
        %Hidden Layer #1 
        X = [X;1*ones(1,batch_size)]; 
        U1 = [W_hidden1 bias_hidden1]*X; 
        H1 = tanh(U1); 

        %Hidden Layer #2 
        U2 = [W_hidden2 bias_hidden2]*[H1;1*ones(1,batch_size)]; 
        H2 = tanh(U2);  
        
        %Output Layer 
        V = [W_output bias_output]*[H2;1*ones(1,batch_size)];
        Y = tanh(V);

        %BACK PROPAGATION
        
        %Desired Output
        classes = y_train(X_indices);
        D = zeros(class_no,batch_size);
        D(:,classes == 1) = ones(class_no,sum(classes == 1)).*[1;-1]; %car neuron = 1, cat neuron = -1 
        D(:,classes == -1) = ones(class_no,sum(classes == -1)).*[-1;1]; %car neuron = -1, cat neuron = 1

        %LOCAL GRADIENT OF OUTPUT LAYER 
        error_output = D-Y; 
        der_of_Y_with_V = 1-Y.^2;      
        delta_output = error_output.*der_of_Y_with_V;

        %LOCAL GRADIENT OF HIDDEN LAYER #2 
        der_of_H2_with_U2 = 1-H2.^2;  
        error_hidden2 = W_output'*delta_output; 
        delta_hidden2 = error_hidden2.*der_of_H2_with_U2; 

        %LOCAL GRADIENT OF HIDDEN LAYER #1 
        der_of_H1_with_U1 = 1-H1.^2;  
        error_hidden1 = W_hidden2'*delta_hidden2; 
        delta_hidden1 = error_hidden1.*der_of_H1_with_U1; 

        %OUTPUT LAYER WEIGHT UPDATE
        delta_We_output =  lr*delta_output*[H2;-1*ones(1,batch_size)]'/batch_size + alpha*delta_We_output;       
        delta_We_hidden2 = lr*delta_hidden2*[H1;-1*ones(1,batch_size)]'/batch_size + alpha*delta_We_hidden2;    
        delta_We_hidden1 = lr*delta_hidden1*X'/batch_size + alpha*delta_We_hidden1; 


        %WEIGHT UPDATES 
        We_output = [W_output bias_output] + delta_We_output; 
        W_output = We_output(:,1:end-1); bias_output = We_output(:,end);

        We_hidden2 = [W_hidden2 bias_hidden2] + delta_We_hidden2;      
        W_hidden2 = We_hidden2(:,1:end-1); bias_hidden2 = We_hidden2(:,end);

        We_hidden1 = [W_hidden1 bias_hidden1] + delta_We_hidden1;         
        W_hidden1 = We_hidden1(:,1:end-1); bias_hidden1 = We_hidden1(:,end);

    end
    
    %FORWARD PROPAGATION FOR TRAIN SET 
    X = [X_train;1*ones(1,train_no)]; 
    U1 = [W_hidden1 bias_hidden1]*X; 
    H1 = tanh(U1);
    U2 = [W_hidden2 bias_hidden2]*[H1;1*ones(1,train_no)];
    H2 = tanh(U2);
    V = [W_output bias_output]*[H2;1*ones(1,train_no)]; 
    Y = tanh(V); 
    
    %ERROR METRICS 
    %Desired Output
    classes = y_train;
    D = zeros(class_no,train_no);
    D(:,classes == 1) = ones(class_no,sum(classes == 1)).*[1;-1]; %car neuron = 1, cat neuron = -1 
    D(:,classes == -1) = ones(class_no,sum(classes == -1)).*[-1;1]; %car neuron = -1, cat neuron = 1
     
    %Mean Squared Error 
    MSE = sum(sum(0.5*(D-Y).^2))/train_no;
    
    %Mean Classification Error 
    [~,real_classes] = max(D);
    [~,pred_classes] = max(Y);   
    MCE = sum(real_classes ~= pred_classes)/train_no * 100;

    history.train_MSE(N) = MSE;
    history.train_MCE(N) = MCE;
                 
    %FORWARD PROPAGATION FOR TEST SET 
    X = [X_test;1*ones(1,test_no)]; 
    U1 = [W_hidden1 bias_hidden1]*X; 
    H1 = tanh(U1);
    U2 = [W_hidden2 bias_hidden2]*[H1;1*ones(1,test_no)];
    H2 = tanh(U2);
    V = [W_output bias_output]*[H2;1*ones(1,test_no)]; 
    Y = tanh(V);
        
    %ERROR METRICS
    %Desired Output
    classes = y_test;
    D = zeros(class_no,test_no);
    D(:,classes == 1) = ones(class_no,sum(classes == 1)).*[1;-1]; %car neuron = 1, cat neuron = -1 
    D(:,classes == -1) = ones(class_no,sum(classes == -1)).*[-1;1]; %car neuron = -1, cat neuron = 1
            
    [~,real_classes] = max(D);
    [~,pred_classes] = max(Y);           
                   
    MSE = sum(sum(0.5*(D-Y).^2))/test_no;
    MCE = sum(real_classes ~= pred_classes)/test_no * 100;
                
    history.test_MSE(N) = MSE;
    history.test_MCE(N) = MCE;
    
end 

end

function [history,weights] = neural_net_Q2 (trainx,traind,valx,vald,testx,testd,epoch_no,batch_size,lr,alpha,D,P)
   
    %Take number of sets 
    train_no = size(traind,1);
    val_no = size(vald,1);
    word_no = max(traind);
    
    %Vectorize words 
    [X1_train,X2_train,X3_train,D_out_train] = prepare_words(trainx,traind);
    [X1_val,X2_val,X3_val,D_out_val] = prepare_words(valx,vald);
        
    %Initialize error metrics for train, validation and test sets 
    history.train_CE = [];
    history.train_MCE = [];
    history.val_CE = [];
    history.val_MCE = [];
    history.test_CE = [];
    history.test_MCE = [];
                
    batch_no = floor(train_no/batch_size);
    
    %EMBEDDING WORDS 
    std  = 0.01;
    WE = std*randn(D,word_no); %embedding matrix 
    W1 = std*randn(P,3*D + 1); %weight matrix of hidden layer #1 
    W2 = std*randn(word_no,P+1); %weight matrix of output layer 
     
    lambda = 1; %parameter of sigmoid activation function 
    delta_W2 = 0;
    delta_W1 = 0;
    delta_WE1 = 0;
    delta_WE2 = 0;
    delta_WE3 = 0;
    
    for i=1:epoch_no  
        
        %shuffle images 
        indices = 1:train_no;
        indices = indices(randperm(train_no)); 

        CE = 0; 
        false_pred_no = 0;
        
        for m = 1:batch_no 
            
            X_indices = indices((m-1)*batch_size+1:m*batch_size);
            X1 = X1_train(:,X_indices);
            X2 = X2_train(:,X_indices);
            X3 = X3_train(:,X_indices);
            D_out = D_out_train(:,X_indices);
            
            %EMBEDDING WORDS 
            E1 = WE*X1;
            E2 = WE*X2;
            E3 = WE*X3;                 

            %CONCATENATED EMBEDDED WORDS 
            E = [E1;E2;E3;ones(1,batch_size)]; %with bias term  
            
            %HIDDEN LAYER #1 
            V1 = W1*E; 
            [Y1,der_Y1] = sigmoid(V1,lambda);

            %HIDDEN LAYER #2 
            Y1 = [Y1;ones(1,batch_size)]; %with bias term 
            V2 = W2*Y1;
            Y2 = softmax(V2);

            %derivative of cross entropy 
            der_CE = -(Y2-D_out); 

            %BACK PROPAGATION 
            %OUTPUT LAYER 
            delta_output = der_CE;

            %HIDDEN LAYER 
            error_hidden = W2(:,1:end-1)'*delta_output;
            delta_hidden = error_hidden.*der_Y1;

            %EMBEDDING MATRIX 
            delta_embed = W1(:,1:end-1)'*delta_hidden;
            delta_embed1 = delta_embed(1:D,:);
            delta_embed2 = delta_embed(D+1:2*D,:);
            delta_embed3 = delta_embed(2*D+1:end,:);

            %UPDATE WEIGHTS  
            delta_W2 = lr*delta_output*Y1'/batch_size + alpha*delta_W2;
            delta_W1 = lr*delta_hidden*E'/batch_size +  alpha*delta_W1;
            delta_WE1 = lr*delta_embed1*X1'/batch_size + alpha*delta_WE1;       
            delta_WE2 = lr*delta_embed2*X2'/batch_size + alpha*delta_WE2;
            delta_WE3 = lr*delta_embed3*X3'/batch_size + alpha*delta_WE3;
            delta_WE = (delta_WE1 + delta_WE2 + delta_WE3)/3;  

            %ERROR METRICS 
            CE = CE + cross_entropy(D_out,Y2);%y = d, y_hat = y2
            [~,real_words] = max(D_out);
            [~,pred_words] = max(Y2);  
            false_pred_no = false_pred_no + sum(real_words ~= pred_words);
           
            %UPDATE WEIGHTS 
            W2 = W2 + delta_W2;
            W1 = W1 + delta_W1;    
            WE = WE + delta_WE;       
             
        end 
        
        CE = sum(CE)/(batch_no * batch_size); 
        MCE = false_pred_no/(batch_no * batch_size);
        history.train_CE = [history.train_CE CE];
        history.train_MCE = [history.train_MCE MCE];

        %VALIDATION SET 
        E1 = WE*X1_val;
        E2 = WE*X2_val;
        E3 = WE*X3_val;     
        E = [E1;E2;E3;ones(1,val_no)]; %bias term  
        V1 = W1*E; 
        [Y1,~] = sigmoid(V1,lambda);
        Y1 = [Y1;ones(1,val_no)]; %bias term 
        V2 = W2*Y1;
        Y2 = softmax(V2);        
        [~,real_words] = max(D_out_val);
        [~,pred_words] = max(Y2); 
        
        %ERROR METRICS 
        false_pred_no = sum(real_words ~= pred_words);
        CE = cross_entropy(D_out_val,Y2);%y = d, y_hat = y2
        CE = sum(CE)/val_no;
        MCE = false_pred_no/val_no;        
        history.val_CE = [history.val_CE CE];
        history.val_MCE = [history.val_MCE MCE];       
        
        if(history.train_CE(end) <= 3)
            break;            
        end 
        
    end 
        
    history.epoch = i;
    
    weights.WE = WE;
    weights.W1 = W1;
    weights.W2 = W2;
    
end 

   


