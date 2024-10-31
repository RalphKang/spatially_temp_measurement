% function blender_hpc_mod2()
% parpool(8);
clc
clear all
%% load data
data=load('./dataset/train_data.txt');
target_ori=load('./dataset/train_temp_normalized.txt');
%%
rng(1); % set seed
order=randperm(size(target_ori, 1)); 
trainpart=0.85;
val_index=order(floor(trainpart*size(target_ori,1))+1:end);
data_val=data(val_index,:);
target=target_ori(val_index,:);
%% normalization
data_max=max(data,[],1);
data_min=min(data,[],1);
data_norm=2.*(data_val-data_min)./(data_max-data_min)-1;

data_max2=max(data_max);
data_min2=min(data_min);
data_rbf=(data_val-data_min2)./(data_max2-data_min2);
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
    y_pred_comb=[yhat_mlp(:,col),yhat_mlp(:,col),yhat,ones(size(target,1),1)];
    %% train second order model
    inverse_matrix=pinv(y_pred_comb'*y_pred_comb);
    coef(:,col)=inverse_matrix*y_pred_comb'*target(:,col);
%     coef(:,col) = regress(target(:,col),y_pred_comb) ;
    y_pred_meta(:,col)=y_pred_comb*coef(:,col);
    col 

end
%%
mse_final=immse(y_pred_meta,target)
writematrix(coef,'linear_meta_model.txt')
    
