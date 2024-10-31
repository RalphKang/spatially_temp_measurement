function svr(dir1,dir2,output1,col,timelimit)
feat=load(dir1);
temp=load(dir2);

%% normalization
feat_max=max(feat,[],1);
feat_min=min(feat,[],1);
features=2.*(feat-feat_min)./(feat_max-feat_min)-1;

%%
rng(1); % set seed
order=randperm(size(temp, 1)); 
trainpart=0.85;
train_index=order(1:round(trainpart*size(temp,1)));
val_index=order(round(trainpart*size(temp,1))+1:end);

x = features;
t = temp;
trainTargets = t(train_index,:);
testTargets = t(val_index,:);
traindata=x(train_index,:);
testdata=x(val_index,:);
%% svr
model = fitrsvm(traindata,trainTargets(:,col),...
    'KernelFunction','polynomial','KernelScale','auto',...
    'Standardize',true,...
    'CacheSize','maximal',...
    'OptimizeHyperparameters',{'BoxConstraint','KernelScale','Epsilon','PolynomialOrder'},...
    'HyperparameterOptimizationOptions',struct('MaxTime',timelimit,'Holdout',0.1));
model_name=strcat('col_',num2str(col),'.mat');
%%
y_hat_test=predict(model,testdata);
mse_test=immse(y_hat_test,testTargets(:,col));
y_hat_train=predict(model,traindata);
mse_train=immse(y_hat_train,trainTargets(:,col));
mse_cb=[mse_train,mse_test];

save(model_name,'model')
%fid = fopen('mse.txt','a');
fid = fopen(output1,'a');
fprintf(fid,'%d \t ',mse_cb); 
fprintf(fid,'\r\n');  % change row
fclose(fid);
%%