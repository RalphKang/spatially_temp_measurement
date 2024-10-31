clc
clear all
% used to pick model file from the saved matlab file
%% cycle reading mse file
for i=1:11
    file_name=strcat('perform/perform_col_',num2str(i),'.txt');
    mse_ori=load(file_name);
    test_mse=mse_ori(:,end);
    [value,index]=min(test_mse);
    mse_sum(i,:)=mse_ori(index,:);
    model_name=strcat('net_summary/col_',num2str(i),'_',num2str(index*2),'.mat');
    model_ori=load(model_name);
    net=model_ori.net;
    new_model_name=strcat('potential_model/col_',num2str(i),'.mat');
    save(new_model_name,'net');
end
%%
mse_mean=mean(mse_sum,1);
%%
writematrix(mse_sum,'mse_blender_sum.txt');
writematrix(mse_mean,'mse_blender_mean.txt');