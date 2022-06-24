function berfin_kavsut_21602459_hw1(question)
clc
close all

switch question
    case '1'
	disp('1')
    disp('Answer is on the report.')
        
    case '2'
	disp('2')    
    disp('Neural Network with Logic Gates')
    
    disp('Part A')    
    disp('Answer is on the report.')
    
    disp('Part B')       
    disp('Bias terms and extended weight matrix of hidden layer:')
    %set weights for each of four hidden units 
    %weights and biases are taken from part a 
    w1 = [1 0 1 1]; b1 = 2.1
    w2 = [0 -1 1 1]; b2 = 1.1
    w3 = [-1 1 -1 0]; b3 = 0.1
    w4 = [-1 1 0 -1]; b4 = 0.1
    
    %single hidden layer for AND implementations  
    %weight matrix with bias terms 
    W1e = [[w1 b1];[w2 b2];[w3 b3];[w4 b4]] %4x5
    
    disp('Bias term and extended weight vector of output layer:')
   
    %output layer(neuron) for OR implementation 
    %weight vector with bias term 
    w5 = [1 1 1 1]; b5 = 0.2
    W2e = [w5 b5]
    
    %all possible inputs with -1 input (bias input)
    input_no = 16; 
    X1 =[0 0 0 0;
         0 0 0 1;
         0 0 1 0;
         0 0 1 1;
         0 1 0 0;
         0 1 0 1;
         0 1 1 0;
         0 1 1 1;
         1 0 0 0;
         1 0 0 1;
         1 0 1 0;
         1 0 1 1;
         1 1 0 0;
         1 1 0 1;
         1 1 1 0;
         1 1 1 1]';
    B = -1*ones(1,input_no);
    X1e = [X1;B]; %5x16
    
    %output vector of first hidden layer 
    V1 = W1e*X1e; %4x16
    O1 = step_activation(V1,0); %4x16
    
    %output of neural network, output from output neuron 
    V2 = W2e*[O1;-1*ones(1,input_no)];
    O2 = step_activation(V2,0);
    o = O2;
    
    %desired output 
    d = logic_gates(X1e);
    
    accuracy = sum(d==o)/input_no*100;
    %disp(strcat('Accuracy percentage:',num2str(accuracy),'% '))
    
    if(accuracy == 100)
        disp('Neural Network works with 100% accuracy!')
    else
        disp('Neural Network does not work with full accuracy!')
    end 
    
    w1e = [w1 b1]; 
    v1 = w1e*X1e;
    o1 = step_activation(v1,0); 
    
    %desired output of neuron #1 
    d = logic_gates_neuron(X1e);
    
    accuracy = sum(d==o1)/input_no*100;
    %disp(strcat('Accuracy percentage:',num2str(accuracy),'% '))
    
    if(accuracy == 100)
        disp('Neuron #1 works with 100% accuracy!')
    else
        disp('Neuron #1 does not work with full accuracy!')
    end 
    
    disp('Part C')
    disp('Bias terms are revised for robustness!');
    %set weights for each of four hidden units 
    %weights and biases are chosen for part c 
    %biases are not selected at the edge of inequalities like part a, 
    %they are in the middle of their min and max values for robustness 
   
    %single hidden layer for AND implementations  
    %weight matrix with bias terms 
    b1_new = 2.5
    b2_new = 1.5
    b3_new = 0.5
    b4_new = 0.5
    W1e_new = [[w1 b1_new];[w2 b2_new];[w3 b3_new];[w4 b4_new]]
    
    %output layer(neuron) for OR implementation 
    %weight vector with bias term 
    b5_new = 0.5
    W2e_new = [w5 b5_new]
    
    disp('Part D')

    %create 25 replicas for each input vector 
    X1e_old = X1e;
    X1e=zeros(5,400);
    for i=1:16
        %25 replicas of each input vector  
        X1e(:,(i-1)*25+1:(i*25)) = repmat(X1e_old(:,i),1,25); 
    end 
    
    N1e = 0 + 0.2*randn(4,400); %mean=0, std=0.2
    N1e = [N1e; zeros(1,400)]; %5x400
    
    %linear combination of random input and noise 
    X1e_noise = X1e + N1e;
    
    %part a 
    V1 = W1e*X1e_noise; 
    O1 = step_activation(V1,0);
    V2 = W2e*[O1;-1*ones(1,400)];
    O2 = step_activation(V2,0);
    O = O2; %1x400
     
    %part c 
    V1 = W1e_new*X1e_noise;
    O1 = step_activation(V1,0);
    V2 = W2e_new*[O1;-1*ones(1,400)];
    O2 = step_activation(V2,0);
    O_new = O2; %1x400
    
    %desired output 
    V1 = W1e*X1e;
    O1 = step_activation(V1,0);
    V2 = W2e*[O1;-1*ones(1,400)];
    O2 = step_activation(V2,0);
    D = O2; %1x400
     
    disp(strcat('Accuracy of neural network in part a: ',num2str(sum(O==D)/400*100),'%'));
    disp(strcat('Accuracy of neural network in part c: ',num2str(sum(O_new==D)/400*100),'%'));    
    
    
    case '3'
	disp('3')
    disp('Perceptron of Alphabet Letters')
 
    disp('Part A')
     
    %read dataset
    dataset = h5readData();
    train_images = im2double(dataset.trainims);   
    test_images = im2double(dataset.testims);
    test_labels = dataset.testlbls; 
    train_labels =  dataset.trainlbls;
   
    %take size of images, training image no, test image no
    [m,l,train_no] = size(train_images);
    [~,~,test_no] = size(test_images);
    class_no = 26; %class_no = unique(train_labels);
    
    %visualize a sample image for each classs
    sample_train = zeros(28,28,26);
    sample_train2 = zeros(28,28,26); %to be used in correlation matrix calculation
    sample_test = zeros(28,28,26);
    
    disp('Sample images from each class are displayed in figure.')
    figure; 
    for i = 1:class_no        
        
        %take indices vector of ith class images 
        index = find(i == train_labels);
        
        %take one random index from ith class, take the train image 
        rand_ind = floor((length(index)-1)*rand()+1);
        sample_train(:,:,i) = train_images(:,:,index(rand_ind));
        
        %take another random index from ith class, take the train image
        rand_ind = floor((length(index)-1)*rand()+1);
        sample_train2(:,:,i) = train_images(:,:,index(rand_ind));
     
        %take one random index from ith class, take the test image
        index = find(i == test_labels);
        rand_ind = floor((length(index)-1)*rand()+1);
        sample_test(:,:,i) = test_images(:,:,index(rand_ind));
        
        %display sample images 
        subplot(5,6,i);
        imshow(squeeze(sample_train(:,:,i)),[]);        
        title(strcat('Class #',num2str(i)));  
        
    end     
    %saveas(gcf,'Q3_Sampled_Images.png');
    
    %correlation matrix 
    %size of (class no x class no)
    %diagonal entries are for within-class correlation coefficients 
    %non-diagnoal entries are for across-class correlation coefficients 
    %p = cor[X,Y] = cov[X,Y]/sqrt((var[X]*var[Y]))
    cor_matrix = zeros(26,26); 
    for i=1:class_no
        for j = 1:class_no
            
            %take one sample images 
            X = squeeze(sample_train(:,:,i));
            Y = squeeze(sample_train(:,:,j));%for across-class
            
            if (i == j)
                Y =  squeeze(sample_train2(:,:,j)); %for within-class 
            end
            
            %find correlation coefficients for X and Y, 
            %which are turned into column vectors 
            R = corrcoef(X,Y);
            cor_matrix(i,j) = R(1,2); %take non-diagonal entry 
         
        end
    end 
    
    disp('Correlation Matrix:')
    cor_matrix
     
    disp('Correlation Matrix is also displayed as an 26x26 image in figure.')
    corr_flag = 1;
    if(corr_flag)
        figure;
        %imshow(cor_matrix,[]);
        imshow(imresize(cor_matrix,[260,260]),[]);
        title('Correlation Matrix');
        %saveas(gcf,'Q3_Correlation_Matrix_resize.png');
% 
%         figure;
%         imshow(cor_matrix,[]);
%         title('Correlation Matrix');    
%         saveas(gcf,'Q3_Correlation_Matrix.png');
    end
    
    %we have neurons as much as class number 
    neuron_no = class_no;
    input_size = m*l; %length of vectorized images 
    
    lambda = 1; %sigmoid function constant  
    n_opt = 0.2; %learning rate 
    
    %take learning rates a lot smaller and a lot bigger than optimum learning rate 
    n = [0.01 1 100]*n_opt;
    
    for k=1:3
        
        %random weights and bias terms from gaussian distribution
        %with zero mean, 0.01 variance
        mean = 0; std = sqrt(0.01);
        W = mean + std*randn(neuron_no,input_size); 
        b = mean + std*randn(neuron_no,1);
        We = [W b]; %extended weight matrix 

        %activation function is sigmoid function 
        %sigmoid_activation(v,lambda,T)

        %start iteration for training model
        MSE = 0;
        iter_no = 10000; 

        for i =1:iter_no

            %take random train image
            rand_ind = floor((train_no-1)*rand()+1);
            x = train_images(:,:,rand_ind);
            x = reshape(x,input_size,1); 
            x = x./(max(x(:))); %rescale image
            xe = [x;-1]; %vectorize image 

            %linear activation potential 
            v = We*xe;
            %it is followed by activation function 
            o = sigmoid_activation(v,lambda,0);
            d =  zeros(class_no,1);
            d(train_labels(rand_ind))= 1;

            %gradient descent update 
            delta_W = n(k)*lambda*(d-o).*(o.*(1-o))*xe';
            We = We + delta_W;

            MSE = (1/class_no)*sum((d-o).^2);
            %objective = (1/2)*norm(real_output-output,2);
            history(k).MSE(i) = MSE; 
            history(k).W(:,:,i) = We(:,1:end-1);
            history(k).b(:,i) = We(:,end);
            %history.objective_function(i) = objective; 

        end 

    end
    
%     for k =1:3
%         figure;
%         for i = 1:class_no  
%             subplot(5,6,i);
%             imshow(reshape(history(k).W(i,:,end),m,l),[]);      
%             title(strcat('Class #',num2str(i)));          
%         end
%         saveas(gcf,strcat('Q3_Trained Weights_',num2str(n(k)),'.png')); 
%     end

    disp('Part B')
    disp('Weights for optimum learning rate are displayed in figure.')
    disp('Optimum learning rate is 0.2.')
   
    %for optimum learning rate, display weights 
    figure;
    for i = 1:class_no  
        subplot(5,6,i);
        imshow(reshape(history(2).W(i,:,end),m,l),[]);      
        title(strcat('Class #',num2str(i)));          
    end
    %saveas(gcf,strcat('Q3_Trained Weights_last_0.2.png')); 
    
    disp('Part C')
    disp('MSE curves are displayed in figure.')
    figure; 
    for k=1:3
        subplot(3,1,k);
        plot(1:iter_no,history(k).MSE);
        title(strcat('MSE, {\eta} = ',num2str(n(k))));
       
        disp(strcat('Learning Rate: ',num2str(n(k))));
        disp(strcat('Last Reached MSE Value:', num2str(history(k).MSE(end))));
   
    end 
    %saveas(gcf,strcat('Q3_MSE_learning rate_0.2.png')); 
    
    disp('Part D')
    disp('Accuracy percentages with different learning rates:')
    
    for k=1:3
        We = [history(k).W(:,:,end) history(k).b(:,end)];
        X = reshape(test_images,input_size,test_no);
        X = X./max(X); %rescale image
        Xe = [X;-1*ones(1,test_no)];
        V = We*Xe;
        O = sigmoid_activation(V,lambda,0);
        [~,output] = max(O);
        D = test_labels'; 
        accuracy = sum(output==D)/test_no*100;

        disp(strcat('Learning Rate: ',num2str(n(k))));
        disp(strcat('Accuracy Percentage: ', num2str(accuracy),'%'));
    end 
    
    case '4'
	disp('4')
    disp('Answer is on the report.')
             
end

end

function output = logic_gates(X)
    %implmeentation of logic gates 
    %output = (X1 OR NOT X2) XOR (NOT X3 OR NOT X4,
    %which is quivalent to ((X1 OR NOT X2) AND NOT (NOT X3 OR NOT X4)) ...
    %OR (NOT (X1 OR NOT X2) AND (NOT X3 OR NOT X4))
    
    %input
    %X: each column vector is one input vector, 
    %contains concatenated input vector 
    
    %implement logic gates for each column vector inside for-loop    
    [row,col] = size(X); 
    output = zeros(1,col);
    for i=1:col
        x1 = X(1,i);
        x2 = X(2,i);
        x3 = X(3,i);
        x4 = X(4,i); 
        output(i) = ((x1||~x2)&~(~x3||~x4)) || (~(x1||~x2)&(~x3||~x4));
    end
    
end
function output = logic_gates_neuron(X)
    %implmeentation of logic gates 
    %output = (X1 OR NOT X2) XOR (NOT X3 OR NOT X4,
    %which is quivalent to ((X1 OR NOT X2) AND NOT (NOT X3 OR NOT X4)) ...
    %OR (NOT (X1 OR NOT X2) AND (NOT X3 OR NOT X4))
    
    %input
    %X: each column vector is one input vector, 
    %contains concatenated input vector 
    
    %implement logic gates for each column vector inside for-loop    
    [row,col] = size(X); 
    output = zeros(1,col);
    for i=1:col
        x1 = X(1,i);
        x3 = X(3,i);
        x4 = X(4,i); 
        output(i) = (x1&x3&x4);
    end
    
end

function result = step_activation(v,T)
    %unipolar step activation function 
    %inputs
    %x: input, T: threshold 
    result = (v >= T);
end

function result = sigmoid_activation(v,lambda,T)
    %unipolar sigmoid activation function 
    %results are between 0-1
    %inputs
    %v: inputs, lambda: parameter for sigmoid, T: threshold 
    if(lambda > 0)
        result = 1./(1+exp(-lambda*(v-T)));
    else
        result = nan;
        disp('Lambda should be a positive value!');
    end        
    
end

function dataset = h5readData()
    %read dataset from hdf5 file 
    
    %h5disp(filename);
    %h5info(filename);
    %h5read(filename);    
    dataset.testims = h5read('assign1_data1.h5','/testims');
    dataset.testlbls = h5read('assign1_data1.h5','/testlbls');
    dataset.trainims = h5read('assign1_data1.h5','/trainims');
    dataset.trainlbls = h5read('assign1_data1.h5','/trainlbls');
end 



