function GPR_opt(location1,location2,feature_number,timelimit)
%% load data
% spect_coef=load("smt_fitting_50_cubic.csv");
spect_coef1=load(location1);
spect_coef=spect_coef1(:,1:feature_number);
temp=load(location2);
spect_max=max(spect_coef,[],1);
spect_min=min(spect_coef,[],1);
spect_coef=2*(spect_coef-spect_min)./(spect_max-spect_min)-1;
%% split the dataset
rng(1); % set seed
data_set=[spect_coef,temp];
order=randperm(size(data_set, 1)); 
rdm_data=data_set(order,:);
train_set_amount=int32(0.85*size(data_set,1));
train_set=rdm_data(1:train_set_amount,:);
train_data=train_set(:,1:size(spect_coef,2));
train_label=train_set(:,size(spect_coef,2)+1:end);
test_set=rdm_data(train_set_amount+1:size(data_set,1),:);
test_data=test_set(:,1:size(spect_coef,2));
test_label=test_set(:,size(spect_coef,2)+1:end);
%%
perform=zeros(size(temp,2),5);
for sub_column=1:size(temp,2) % each subcolumns
        %% five models are used        
        gprMd= fitrgp(train_data,train_label(:,sub_column),'KernelFunction','matern52',...
            'OptimizeHyperparameters',{'KernelScale','Sigma'},...
    'HyperparameterOptimizationOptions',struct('MaxTime',timelimit,'Holdout',0.1));   
    model_name=strcat('col_',num2str(sub_column),'.mat');
    save(model_name,'gprMd');
        %% evalue model
        y_predic=predict(gprMd,test_data);
        y_hat=y_predic;
        y_predic_train=predict(gprMd,train_data);
        y_hat_train=y_predic_train;
        
        R=corrcoef(test_label(:,sub_column),y_hat);
        r2=R(2,1);
        mse=immse(test_label(:,sub_column),y_hat);
        R_train=corrcoef(train_label(:,sub_column),y_hat_train);
        r2_train=R_train(2,1);
        mse_train=immse(train_label(:,sub_column),y_hat_train);
    perform(sub_column,:)=[sub_column,r2_train,mse_train,r2,mse];
    fid = fopen('column_record.txt','a');
    fprintf(fid,'%d \t ',perform(sub_column,:)); 
    fprintf(fid,'\r\n');  % »»ÐÐ
    fclose(fid);
end
perform_mean=mean(perform,1);
writematrix(perform_mean,'perform.csv')