% function blender_hpc_vali()
% parpool(2)
%% load data
data=load('./dataset/train_data.txt');
data_test=load('./test_dataset/test_data.txt');
target=load('./test_dataset/test_temp_normalized.txt');
coef=load('linear_meta_model.txt'); % linear model
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
    y_pred_comb=[yhat_mlp(:,col),yhat_mlp(:,col),yhat,ones(size(target,1),1)];
    %% train second order model
    yhat_final(:,col)=y_pred_comb*coef(:,col);
  col  
end
%%
dlmwrite('regression_linear_blender.txt',yhat_final)
mean_mse=immse(yhat_final,target);
yhat_res=reshape(yhat_final,size(yhat_final,1)*size(yhat_final,2),1);
target_res=reshape(target,size(target,1)*size(target,2),1);
%%
R = corrcoef(yhat_res,target_res);
r=R(2,1);
dlmwrite('meta_mse.txt',mean_mse);
dlmwrite('meta_R.txt',r);
    
