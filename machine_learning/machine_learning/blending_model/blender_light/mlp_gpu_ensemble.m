function mlp_gpu_ensemble(features,temp,col,max_neuron)
dir3=strcat('record/record_col_',num2str(col),'.txt');
dir4=strcat('perform/perform_col_',num2str(col),'.txt');
dir5=strcat('pred_test/pred_test_col_',num2str(col),'.txt');
dir6=strcat('pred_train/pred_train_col_',num2str(col),'.txt');
%% data loading
% features=load("linear_150.txt"); %load spectrum data (processed)
% temperature=load("temperature_norm_11D_col.txt");% load temperature data(processed)

%%
rng(1); % set seed
order=randperm(size(temp, 1)); 
trainpart=0.85;
train_index=order(floor(trainpart*size(temp,1))+1:floor(trainpart*size(temp,1))+3000);
val_index=order(floor(trainpart*size(temp,1))+3001:end);

x = features';
t = temp';
trainTargets = t(:,train_index);
valTargets = t(:,val_index);
traindata=x(:,train_index);
testdata=x(:,val_index);
%% build net structure 
all_output=zeros(max_neuron,5);
for neurons=2:2:max_neuron
    trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
    
    total_turns=5;
    r_train=zeros(1,total_turns);
    r_test=zeros(1,total_turns);
    mse_train=ones(1,total_turns);
    mse_test=ones(1,total_turns);
    for round=1:total_turns
        hiddenLayerSize = neurons;
        net = fitnet(hiddenLayerSize,trainFcn);

        net.input.processFcns = {'removeconstantrows'};
        net.output.processFcns = {'removeconstantrows'};
        
        net.divideFcn = 'divideind';
        net.divideParam.trainInd=train_index;
        net.divideParam.valInd=val_index;  
        
        net.trainParam.show=10;
        net.trainParam.showCommandLine=true;
        net.trainParam.epochs=3000;
        net.trainParam.max_fail=6;
        net.performFcn = 'mse';  % Mean Squared Error
        %% train and test network
        [net,tr] = train(net,x,t,'useGPU','yes','useParallel','yes');
        pred_train = net(traindata);
        pred_test = net(testdata);
        %% evaluate network
        mse_train(round) = immse(pred_train,trainTargets);
         mse_test(round) = immse(pred_test,valTargets);
        % testPerformance = perform(net,testTargets,y);

%         [r_train(round),mse_train(round)]=rsquare(y,trainTargets);
%         [r_test(round),mse_test(round)]=rsquare(y,valTargets);
        pred_train_res=reshape(pred_train,size(pred_train,1)*size(pred_train,2),1);
        train_res=reshape(trainTargets,size(trainTargets,1)*size(trainTargets,2),1);
        R=corrcoef(pred_train_res,train_res,'rows','complete');
        r_train(round)=R(2,1);
        
        pred_test_res=reshape(pred_test,size(pred_test,1)*size(pred_test,2),1);
        val_res=reshape(valTargets,size(valTargets,1)*size(valTargets,2),1);
        R=corrcoef(pred_test_res,val_res,'rows','complete');
        r_test(round)=R(2,1);
        
        if r_test(round)==max(r_test)
            net_name=strcat('net_summary/col_',num2str(col),'_',num2str(neurons),".mat");
            save(net_name,"net")
            %writematrix(pred_test,dir5)
            %dlmwrite(dir5,pred_test)
            %writematrix(pred_train,dir6)
            %dlmwrite(dir6,pred_train)
        end
        output= [neurons,r_train(round),mse_train(round),r_test(round),mse_test(round)];
        %every round record
        fid = fopen(dir3,'a');
        fprintf(fid,'%d \t ',output); 
        fprintf(fid,'\r\n');  % change row
        fclose(fid);
    end
    %%
    average_perf=mean([r_train;mse_train;r_test;mse_test],2);
    all_output(neurons,:)=[neurons,average_perf'];
    fid = fopen(dir4,'a');
        fprintf(fid,'%d \t ',all_output(neurons,:)); 
        fprintf(fid,'\r\n');  % change row
        fclose(fid);
end
end


%% auxiliary functions 
function [r,mse] = rsquare(y,f,varargin)
% Compute coefficient of determination of data fit model and RMSE
%
% [r2 rmse] = rsquare(y,f)
% [r2 rmse] = rsquare(y,f,c)
%
% RSQUARE computes the coefficient of determination (R-square) value from
% actual data Y and model data F. The code uses a general version of 
% R-square, based on comparing the variability of the estimation errors 
% with the variability of the original values. RSQUARE also outputs the
% root mean squared error (RMSE) for the user's convenience.
%
% Note: RSQUARE ignores comparisons involving NaN values.
% 
% INPUTS
%   Y       : Actual data
%   F       : Model fit
%
% OPTION
%   C       : Constant term in model
%             R-square may be a questionable measure of fit when no
%             constant term is included in the model.
%   [DEFAULT] TRUE : Use traditional R-square computation
%            FALSE : Uses alternate R-square computation for model
%                    without constant term [R2 = 1 - NORM(Y-F)/NORM(Y)]
%
% OUTPUT 
%   R2      : Coefficient of determination
%   RMSE    : Root mean squared error
%
% EXAMPLE
%   x = 0:0.1:10;
%   y = 2.*x + 1 + randn(size(x));
%   p = polyfit(x,y,1);
%   f = polyval(p,x);
%   [r2 rmse] = rsquare(y,f);
%   figure; plot(x,y,'b-');
%   hold on; plot(x,f,'r-');
%   title(strcat(['R2 = ' num2str(r2) '; RMSE = ' num2str(rmse)]))
%   
% Jered R Wells
% 11/17/11
% jered [dot] wells [at] duke [dot] edu
%
% v1.2 (02/14/2012)
%
% Thanks to John D'Errico for useful comments and insight which has helped
% to improve this code. His code POLYFITN was consulted in the inclusion of
% the C-option (REF. File ID: #34765).
if isempty(varargin); c = true; 
elseif length(varargin)>1; error 'Too many input arguments';
elseif ~islogical(varargin{1}); error 'C must be logical (TRUE||FALSE)'
else c = varargin{1}; 
end
% Compare inputs
if ~all(size(y)==size(f)); error 'Y and F must be the same size'; end
% Check for NaN
tmp = ~or(isnan(y),isnan(f));
y = y(tmp);
f = f(tmp);
if c; r2 = max(0,1 - sum((y(:)-f(:)).^2)/sum((y(:)-mean(y(:))).^2));
else r2 = 1 - sum((y(:)-f(:)).^2)/sum((y(:)).^2);
    if r2<0
    % http://web.maths.unsw.edu.au/~adelle/Garvan/Assays/GoodnessOfFit.html
        warning('Consider adding a constant term to your model') %#ok<WNTAG>
        r2 = 0;
    end
end
% rmse = sqrt(mean((y(:) - f(:)).^2));
mse=mean((y(:) - f(:)).^2);
r=sqrt(r2);
end


