% function blender_hpc_vali()
% parpool(2)
% this code is used to test the performance of blender model
%% load data
data=load('./dataset/train_data.csv');
data_test=load('./test_dataset/test_data.csv');
target=load('./test_dataset/test_temp_normalized.txt');
%% normalization
data_max=max(data,[],1);
data_min=min(data,[],1);
data_norm=2.*(data_test-data_min)./(data_max-data_min)-1;

data_max2=max(data_max);
data_min2=min(data_min);
data_rbf=(data_test-data_min2)./(data_max2-data_min2);

%% net model----------------------------------------------
%% mlp
mlp_location='./model/MLP/20.mat';
[yhat_mlp,R_mlp,mse_mlp]=net_validation(data_norm(:,1:500)',target',mlp_location);
yhat_mlp=yhat_mlp';
%% rbf
rbf_location='./model/RBF/rbf_net.mat';
[yhat_rbf,R_rbf,mse_rbf]=net_validation(data_rbf',target',rbf_location);
yhat_rbf=yhat_rbf';
%% single output model
yhat_final=zeros(size(data_test,1),11);
for col=1:11

    model_name=strcat('col_',num2str(col),'.mat');
    name_list={strcat('model/svr_gas/',model_name),strcat('model/svr_linear/',model_name),...
        strcat('model/svr_poly/',model_name),strcat('model/gpr_exp/',model_name),...
        strcat('model/gpr_raq/',model_name),strcat('model/gpr_mat32/',model_name)...
        strcat('model/gpr_mat52/',model_name),strcat('model/gpr_seq/',model_name)};
    feature_size=[540,540,540,300,500,200,200,300];
    model_size=size(name_list,2);
    sample_num=size(data_norm,1);
    %% 
    yhat=zeros(sample_num,model_size);
    for j=1:model_size
    [yhat(:,j),R(j),mse(j)]=model_validation(data_norm(:,1:feature_size(j)),target(:,col),name_list{j});
    end
    %% combine data
    y_pred_comb=[yhat_mlp(:,col),yhat_mlp(:,col),yhat];
    %% train second order model
    meta_learner_location=strcat('potential_model/col_',num2str(col),'.mat');
    [yhat_meta,R_meta(col,j),mse_meta(col,j)]=net_validation(y_pred_comb',target(:,col)',meta_learner_location);
    yhat_final(:,col)=yhat_meta;
  col  
end
dlmwrite('regression_blender.txt',yhat_final)
mean_mse=mean(mse_meta,1);
mean_R=mean(R_meta,1);
dlmwrite('meta_mse.txt',mean_mse);
dlmwrite('meta_R.txt',mean_R);
    
