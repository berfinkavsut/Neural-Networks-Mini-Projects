function berfin_kavsut_21602459_hw3(question)
clc
close all

switch question
    
    case '1'
	disp('Q1')
    
    %Part A 
    
    %read dataset  
    data = h5read('assign3_data1.h5','/data');
    %invXForm = h5read('assign3_data1.h5','/invXForm');    
    %xForm = h5read('assign3_data1.h5','/xForm');
    
    %Graysale using luminosity model: Y = 0.2126*R + 0.7152*G + 0.0722*B
    R = data(:,:,1,:);
    G = data(:,:,2,:);
    B = data(:,:,3,:);
    Y_gray = 0.2126*R + 0.7152*G + 0.0722*B;
    Y_gray = squeeze(Y_gray);
    
    %Demean
    [m,m,N] = size(Y_gray);
    %vectorize images 
    Y_reshaped = reshape(Y_gray,m*m,N); 
    
    %mean of images 
    mean_Y = mean(Y_reshaped,1);
    %subtract mean image from images 
    Y_demean = Y_reshaped - mean_Y;
    
    %Destd
    Y_std = zeros(size(Y_reshaped));
    for i=1:N
        img = Y_demean(:,i);
        std = sqrt(mean(img.^2)); %std = sqrt(var) = sqrt(E[(X-E(X))^2])
        %clip image pixels with more than +/-3 std 
        img(img>(img+3*std)) = img(img>(img+3*std))+3*std;
        img(img<(img-3*std)) = img(img<(img-3*std))-3*std;
        Y_std(:,i) = img;
    end 

    %Normalize and map to [0.1,0.9]
    Y_norm = zeros(size(Y_reshaped));
    epsilon = 1e-10; % for zero-division problem 
    for i=1:N
        img = Y_std(:,i);
        min_val = min(img);
        max_val = max(img);        
        img_norm = (img-min_val)./(max_val-min_val+epsilon);%normalize btw [0,1]
        img_norm = img_norm*0.8+0.1;% map to the interval [0.1,0.9]
        Y_norm(:,i) = img_norm;
    end 
    
    %2-D images 
    Y = reshape(Y_norm,m,m,N);    

    %take random 200 index 
    indices = 1:N;
    indices = indices(randperm(N));    
    indices = indices(1:200);  

    %RGB Images
    %show all images in one big figure 
    for l = 1:5        
        figure;set(gcf, 'WindowState', 'maximized');
        for i =1:40           
            ind = indices((l-1)*40+i);
            subplot(5,8,i);            
            imshow(imresize(data(:,:,:,ind),20,'nearest')); 
            title(strcat('Index:',num2str(ind)));   
        end 
        %saveas(gcf,strcat('RGB_images',num2str(l),'.png'));
    end
         
    %Normalized Images
    %show all images in one big figure 
    for l = 1:5        
        figure;set(gcf, 'WindowState', 'maximized');
        for i =1:40           
            ind = indices((l-1)*40+i);
            subplot(5,8,i);            
            imshow(imresize(Y(:,:,ind),20,'nearest'));
            title(strcat('Index:',num2str(ind)));   
        end 
        %saveas(gcf,strcat('norm_images',num2str(l),'.png'));
    end    
    
    %Part B     
    Y = reshape(Y,m*m,N);
    
    %parameters 
    Lin = m*m;
    Lhid = 64; 
    lambda = 5e-4; 
    
    %optimal beta and rho base on weight images 
    beta = 2;
    rho = 0.005;%typical small value, close to 0 
    
    %interval limits [-wo,wo]
    wo = sqrt(6/(Lin+Lhid));    
    W1 = 2*wo*rand(Lhid,Lin) - wo;
    b1 = 2*wo*rand(Lhid,1) - wo;
    W2 = W1';
    b2 = 2*wo*rand(Lin,1) - wo;
    
    %weights and biases are contained in We struct 
    We.W1 = W1;
    We.W2 = W2;
    We.b1 = b1;    
    We.b2 = b2;   
    
    %parameters for aeCost function 
    params.Lin = Lin; 
    params.Lhid = Lhid; 
    params.lambda = lambda; 
    params.beta = beta;
    params.rho = rho;  
   
    %parameters 
    lr = 0.1;
    epoch_no = 50;
    batch_size = 32; 
    batch_no = N/batch_size;
    
    %train autoencoder 
    loss=zeros(1,epoch_no);    
    for i=1:epoch_no
        error = 0;
        for j = 1:batch_no
            %take random samples as much as mini-batch size 
            indices = 1:N;
            indices = indices(randperm(N));    
            indices = indices((j-1)*batch_size+1: j*batch_size);            
            data = Y(:,indices);
            
            %Gradients
            [J,Jgrad] = aeCost(We,data,params);
            %Gradient Descent Solver 
            [We_update] = grad_descent_solver(J,Jgrad,We,data,lr);
            We = We_update;
            error = error + J;
        end         
        loss(i) = error/batch_no;
    end 
    
    %draw loss function 
    figure; plot(loss);title('Loss Function'); 
    xlabel('Epoch No');ylabel('Loss')
    %saveas(gcf,'loss.png');
    
    %draw weights to understand feature extraction process
    figure;set(gcf, 'WindowState', 'maximized');
    W = We.W1;
    for i=1:Lhid
        w = squeeze(W(i,:));
        w = reshape(w,m,m);
        subplot(sqrt(Lhid),sqrt(Lhid),i);
        imshow(imresize(w,20,'nearest'),[]); 
        title(num2str(i))
    end 
    %saveas(gcf,strcat('rho_',num2str(rho),'_beta_',num2str(beta),'.png'));
    
    %Part D 
    for Lhid = [16,64,100]
        for lambda = [1e-10,5e-4,1e-3] 

            %optimal beta and rho base on weight images 
            beta = 2;
            rho = 0.005; %typical small value, close to 0 

            %weight and bias initialized randomly  
            %interval limits [-wo,wo]
            wo = sqrt(6/(Lin+Lhid));    
            W1 = 2*wo*rand(Lhid,Lin) - wo;
            b1 = 2*wo*rand(Lhid,1) - wo;
            W2 = W1';
            b2 = 2*wo*rand(Lin,1) - wo;

            %weights and biases are in We struct 
            We.W1 = W1;
            We.W2 = W2;
            We.b1 = b1;    
            We.b2 = b2;   

            %parameters for aeCost function are in param struct  
            params.Lin = Lin; 
            params.Lhid = Lhid; 
            params.lambda = lambda; 
            params.beta = beta;
            params.rho = rho;  

            %hyperparameters 
            lr = 0.1;
            epoch_no = 50;
            batch_size = 32; 
            batch_no = N/batch_size;

            %train autoencoder 
            for i=1:epoch_no
                for j = 1:batch_no
                    %take random samples as much as mini-batch size 
                    indices = 1:N;
                    indices = indices(randperm(N));    
                    indices = indices((j-1)*batch_size+1: j*batch_size);            
                    data = Y(:,indices);

                    %Gradients 
                    [J,Jgrad] = aeCost(We,data,params);                    
                    %Gradient Descent Solver 
                    [We_update] = grad_descent_solver(J,Jgrad,We,data,lr);
                    We = We_update;
                end         
            end 
    
            %Draw weights to understand feature extraction process
            figure;set(gcf, 'WindowState', 'maximized');
            W = We.W1;
            for i=1:Lhid
                w = squeeze(W(i,:));
                w = reshape(w,m,m);
                subplot(sqrt(Lhid),sqrt(Lhid),i);
                imshow(imresize(w,20,'nearest'),[]);   
                title(num2str(i));
            end 
            %saveas(gcf,strcat('Lhid_',num2str(Lhid),'_lambda_',num2str(lambda),'.png'));
        end 
    end 
    
    case '3'
    disp('Q3')

    %read dataset  
    trX = h5read('assign3_data3.h5','/trX');
    trY = h5read('assign3_data3.h5','/trY');    
    tstX = h5read('assign3_data3.h5','/tstX');
    tstY = h5read('assign3_data3.h5','/tstY');
 
    %take sizes
    [sensor_no, time_unit, train_no] = size(trX);  
    [class_no,~] = size(trY);
    [sensor_no, time_unit, test_no] = size(tstX);
     
    %NN layer sizes 
    rnn_neuron = 128; %RNN hidden layer neuron number 
    input_size = sensor_no; %one sample is 3x150 matrix 
    hidden_no = 10; %MLP 
    class_no = 6; %MLP   
    
    %hyperparameters 
    lr = 0.01; %learning rate
    alpha = 0.8; %momentum 
    batch_size = 32; 
    epoch_no = 50; %max 50 epoch  
    
    %Weight/Bias Initialization with Xavier Uniform Distribution 
    r = sqrt(6/(input_size+hidden_no));
    Wih = 2*r*rand(rnn_neuron,input_size) - r;
    bih = 2*r*rand(rnn_neuron,1) - r;  
    
    r = sqrt(6/(rnn_neuron+hidden_no));
    Whh = 2*r*rand(rnn_neuron,rnn_neuron) - r;    
    bhh = 2*r*rand(rnn_neuron,1) - r;  
    
    r = sqrt(6/(rnn_neuron+class_no));
    W1 = 2*r*rand(hidden_no,rnn_neuron) - r;
    b1 = 2*r*rand(hidden_no,1) -r; 
    
    r = sqrt(6/(hidden_no+class_no));
    Wo = 2*r*rand(class_no,hidden_no) - r;
    bo = 2*r*rand(class_no,1) - r; 
    
    %error metrics 
    history.val_CE = [];
    history.val_acc = [];
      
    %Validation Set 
    %validation sample number 
    val_no = floor(train_no/10);    
    %take random indices 
    val_ind = 1:train_no;
    val_ind = val_ind(randperm(train_no));        
    %seperate validation set 
    val_X = trX(:,:,val_ind(1:val_no));        
    val_D = trY(:,val_ind(1:val_no));    
    %delete validation set form train set 
    trX(:,:,val_ind(1:val_no)) = [];
    trY(:,val_ind(1:val_no)) = [];  
    train_no = train_no - val_no;    
    
    %mini-batch stochastic gradient descent 
    batch_no = floor(train_no/batch_size);   
   
    for N = 1:epoch_no 

        %weight updates from previous update 
        mdWo = zeros(class_no,hidden_no+1);
        mdW1 = zeros(hidden_no,rnn_neuron+1);        
        mdWih = zeros(rnn_neuron,input_size+1);        
        mdWhh = zeros(rnn_neuron,rnn_neuron+1);
        
        %shuffle training images (shuffle indices)
        indices = 1:train_no;
        indices = indices(randperm(train_no));    
         
        for j = 1:batch_no
          
            %take samples as much as batch size and average them
            x_indices = indices( (j-1)*batch_size+1 : j*batch_size );
            x_matrix = trX(:,:,x_indices);
            x = sum(x_matrix,3)/batch_size; 
            
            %States thorough time units 
            s_prev = zeros(rnn_neuron,1); %initialize first state as zero 
            S = [s_prev]; 
            %Inputs thorough time units 
            X = [];
           
            %FORWARD PROPAGATION
            for i = 1:time_unit                
                
                input = x(:,i); %ith time instance    
                
                %RNN
                s = tanh([Wih bih]*[input;1] + [Whh bhh]*[s_prev;1]);
                
                %Save current input and state for backpropagation 
                X = [X input];
                S = [S s];                
                s_prev = s; %for next forward propagation                   
            end            
            
            %MLP (single hidden layer)
            v1 = [W1 b1]*[s;1]; %take last state as input 
            y1 = reLu(v1); %activation 
            v2 = [Wo bo]*[y1;1];
            o = softmax(v2); %classification result 
        
            %BACKPROPAGATION 
            
            %desired outputs 
            d_matrix = trY(:,x_indices);
            d = sum(d_matrix,2)/batch_size; 
           
            %gradients of MLP 
            do = -(o-d); %derivative of cross entopy with softmax output
            d1 = (Wo'*do).*(y1~=0); %derivative of sigmoid act. func. 
            
            %weight update terms of MLP 
            dWo = do*[y1;1]';
            dW1 = d1*[s;1]';
       
            %gradient of hidden state layer 
            dh = (W1'*d1).*(1-s.^2); % derivative of tanh
            
            S(:,end) = [];
            
            %initialize weight update terms 
            dWih_sum = zeros(rnn_neuron,input_size+1);
            dWhh_sum = zeros(rnn_neuron,rnn_neuron+1);            
            for k = 1:time_unit                      
                %accumulate weight update terms thorough time instances 
                %as going back in time units 
                dWih_sum = dWih_sum + dh*[X(:,end);1]';               
                dWhh_sum = dWhh_sum + dh*[S(:,end);1]';
                
                %gradient of hidden state layer 
                %one time unit back in RNN 
                dh = (Whh'*dh).*(1-S(:,end).^2);
                
                %delete last input and state to go one time unit back 
                X(:,end) = [];
                S(:,end) = []; 
            end 
            
            %weight updates with momentum term 
            mdWo = lr*dWo + alpha*mdWo;             
            mdW1 = lr*dW1 + alpha*mdW1;
            mdWih = lr*dWih_sum + alpha*mdWih;             
            mdWhh = lr*dWhh_sum + alpha*mdWhh;
            
            %updating weights and biases for all network 
            %MLP 
            Wo = Wo + mdWo(:,1:end-1);
            bo = bo + mdWo(:,end);
            W1 = W1 + mdW1(:,1:end-1);
            b1 = b1 + mdW1(:,end);
            %RNN
            Wih = Wih + mdWih(:,1:end-1);
            bih = bih + mdWih(:,end);
            Whh = Whh + mdWhh(:,end-1);    
            bhh = bhh + mdWhh(:,end);
            
        end       
        
        %VALIDATION SET 
        CE = 0;
        true_pred = 0;
        for n = 1:val_no
            
            %FORWARD PROPAGATION
            x = val_X(:,:,n);
            s_prev = zeros(rnn_neuron,1);  
            
            for i = 1:time_unit                
                input = x(:,i); %ith time instance                 
                %RNN
                s = tanh([Wih bih]*[input;1] + [Whh bhh]*[s_prev;1]);                            
                s_prev = s; %for next forward propagation                   
            end   
            
            %MLP
            v1 = [W1 b1]*[s;1];
            y1 = reLu(v1); %single hidden layer of MLP 
            v2 = [Wo bo]*[y1;1];
            o = softmax(v2);  
            
            d = val_D(:,n); %desired output 
            CE = CE + cross_entropy(d,o);
            [~,desired] = max(d);
            [~,pred] = max(o); 
            true_pred = true_pred + (desired == pred);
            
        end     
        CE = CE/val_no; %average error 
        val_acc = true_pred/val_no; %validation accuracy 
        history.val_acc = [history.val_acc val_acc];
        history.val_CE = [history.val_CE CE];  
     
        if (N==1)
            init_err = CE;
            tol_err = init_err - 0.1;
        end 
        
        if(N>1)
            if(CE < tol_err)
               break
            end 
        end 
        
        disp(['Epoch No:',num2str(N),' Validation Error:',num2str(history.val_CE(N))]);
  
    end 
   
%     figure; 
%     plot(history.val_acc);title('Validation Accuracy');
%     xlabel('Epoch No');ylabel('Accuracy');    
%     for i=1:length(history.val_acc)
%         disp(['Epoch No:',num2str(i),' Validation Accuracy:',num2str(history.val_acc(i))]);
%     end 
    
    figure; 
    plot(history.val_CE);title('Validation Error');
    xlabel('Epoch No');ylabel('Error');
    %saveas(gcf,'val_error.png');
    
    %TRAINING SET - CONFUSION MATRIX     
    train_X = trX;        
    train_D = trY;  
    
    %prediction 
    true_pred = 0;
    D = []; %desired output 
    P = []; %predicted output 
    
    for n = 1:train_no
        
        %FORWARD PROPAGATION
        x = train_X(:,:,n);
        s_prev = zeros(rnn_neuron,1);        
        for i = 1:time_unit                
            input = x(:,i); %ith time instance                 
            %RNN
            s = tanh([Wih bih]*[input;1] + [Whh bhh]*[s_prev;1]);                            
            s_prev = s; %for next forward propagation                   
        end            
        %MLP
        v1 = [W1 b1]*[s;1];
        y1 = reLu(v1); %single hidden layer of MLP 
        v2 = [Wo bo]*[y1;1];
        o = softmax(v2);  

        %desired output 
        d = train_D(:,n);
        
        % prediction and real output 
        [~,desired] = max(d);
        [~,pred] = max(o); 
        %save prediction and real output  
        P = [P pred];
        D = [D desired];
        
        %true prediction number  
        true_pred = true_pred + (desired == pred);

    end       
    %train_acc = true_pred/train_no*100;  
    %disp(['Train Accuracy:',num2str(train_acc),'%']);
    
    %Confusion Matrix of Test Set 
    C = confusionmat(D,P);
    figure; confusionchart(C); title('Training Set');
    %saveas(gcf,'confusion_matrix_train.png');
    
    %TEST SET - ACCURACY AND CONFUSION MATRIX     
    test_X = tstX;        
    test_D = tstY;  
    
    %prediction 
    true_pred = 0;
    D = []; %desired output 
    P = []; %predicted output 
    
    for n = 1:test_no
        
        %FORWARD PROPAGATION
        x = test_X(:,:,n);
        s_prev = zeros(rnn_neuron,1);        
        for i = 1:time_unit                
            input = x(:,i); %ith time instance                 
            %RNN
            s = tanh([Wih bih]*[input;1] + [Whh bhh]*[s_prev;1]);                            
            s_prev = s; %for next forward propagation                   
        end            
        %MLP
        v1 = [W1 b1]*[s;1];
        y1 = reLu(v1); %single hidden layer of MLP 
        v2 = [Wo bo]*[y1;1];
        o = softmax(v2);  

        %desired output 
        d = test_D(:,n);
        
        % prediction and real output 
        [~,desired] = max(d);
        [~,pred] = max(o); 
        %save prediction and real output  
        P = [P pred];
        D = [D desired];
        
        %true prediction number  
        true_pred = true_pred + (desired == pred);

    end       
    test_acc = true_pred/test_no*100;  
    disp(['Test Accuracy:',num2str(test_acc),'%'])
    
    %Confusion Matrix of Test Set 
    C = confusionmat(D,P);
    figure; confusionchart(C); title('Test Set')
    %saveas(gcf,'confusion_matrix_test.png');
    

end


end     

function [J,Jgrad] = aeCost(We,data,params)
    
    %sample number 
    m = size(data,1);
    N = size(data,2);
    
    %weights and biases
    W1 = We.W1;
    W2 = We.W2;
    b1 = We.b1;
    b2 = We.b2;
   
    %parameters 
    Lin = params.Lin;
    Lhid = params.Lhid;
    lambda = params.lambda; 
    beta = params.beta;
    rho = params.rho;
    
    %FORWARD PROPAGATION 
    x = data;
    d = x; 
    v = [W1 b1]*[x;ones(1,N)];    
    a = sigmoid(v,1);
    o = [W2 b2]*[a;ones(1,N)];
    
    %Average Squared Error
    ASE = (1/2)*sum(sum((d/sqrt(N)-o/sqrt(N)).^2));%squared error
     
    %Tykhonov Regularization 
    tykhonov_reg = (lambda/2)*(sum(W1(:).^2) + sum(W2(:).^2));
    
    %Kullback-Leibler (KL)  
    
    %average activation of hidden unit b 
    x_sum = sum(x,1); %1xN
    rho_sum = x_sum.*a/(m); %LhidxN
    rho_b  = sum(rho_sum,2)/N; %take average, Lhidx1
   
    %KL term     
    KL = rho.*log(rho./rho_b)+(1-rho).*log((1-rho)./(1-rho_b)); %Lhidx1
    KL = beta * sum(KL); %1x1
    
    %loss function
    J = ASE + tykhonov_reg + KL;
    
    %gradients 
    do = -(d-o);
    dW1_tyk = lambda*W1;
    dW2_tyk = lambda*W2;
    dhid = ((W2'*do)+ beta*(-rho./rho_b+(1-rho)./(1-rho_b))).*(a.*(1-a)); %64x10240
   
    %save gradients inside Jgrad struct 
    Jgrad.do = do;
    Jgrad.dW1_tyk = dW1_tyk;
    Jgrad.dW2_tyk = dW2_tyk;
    Jgrad.dhid = dhid;    
    
end 

function [We] = grad_descent_solver(J,Jgrad,We,data,lr)
    
    %sample number 
    N = size(data,2);
    
    %weights and biases
    W1 = We.W1;
    W2 = We.W2;
    b1 = We.b1;
    b2 = We.b2;
    
    %gradients 
    do = Jgrad.do;
    dW1_tyk = Jgrad.dW1_tyk;
    dW2_tyk = Jgrad.dW2_tyk;
    dhid = Jgrad.dhid;   
    
    %FORWARD PROPAGATION (SECOND TIME FOR BACKPROPAGATION)
    x = data;
    d = x; %desired output is input 
    v = [W1 b1]*[x;ones(1,N)]; %first layer   
    a = sigmoid(v,1); %activation 
    o = [W2 b2]*[a;ones(1,N)]; %second layer 
    
    %weight and bias update terms 
    dW2 = do*a'/N +dW2_tyk; 
    dW1 = dhid*x'/N +dW1_tyk;
    db2 = sum(do,2)/N;
    db1 = sum(dhid,2)/N;
    
    %update weights and biases 
    W1 = W1 - lr*dW1;
    W2 = W2 - lr*dW2;
    b1 = b1 - lr*db1;
    b2 = b2 - lr*db2;    

    %save them to We stuct
    We.W1 = W1;    
    We.W2 = W2;    
    We.b1 = b1;    
    We.b2 = b2;
    
end 


function o = reLu(x)
    o = max(x,0);
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

function o = sigmoid(v,lambda)
    %Unipolar sigmoid activation function 
    %Results are on interval of [0,1]
    %Inputs
    %v: inputs, lambda: parameter for sigmoid, T: threshold 
    if(lambda > 0)
        o = 1./(1+exp(-lambda*v));
    else
        o = nan;
        disp('Lambda should be a positive value!');
    end     
end

  


