% this code is used to train blender and blender light model
function blender_hpc_mod3()
parpool(8);
%% load data
data=load('./dataset/train_data.txt');
target=load('./dataset/train_normalized_temp.txt');
%% normalization
data_max=max(data,[],1);
data_min=min(data,[],1);
data_norm=2.*(data-data_min)./(data_max-data_min)-1;

data_max2=max(data_max);
data_min2=min(data_min);
data_rbf=(data-data_min2)./(data_max2-data_min2);
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
    y_pred_comb=[yhat_mlp(:,col),yhat_mlp(:,col),yhat];
    %% save new_input
%     new_input_name=strcat('twice_input/twice_input_col_',num2str(col),'.txt');
    %writematrix(y_pred_comb,new_input_name)
    %dlmwrite(new_input_name,y_pred_comb)
    %% train second order model
    mlp_gpu_ensemble(y_pred_comb,target(:,col),col,100)
    col

end
    
