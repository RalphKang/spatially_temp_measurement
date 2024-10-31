% check the performance of the network
function [yhat,r,mse]=net_validation(feature,temp_norm,net_location)
%%
net=load(net_location);
real_net=net.net;
yhat=real_net(feature);

%%
temp_norm_res=reshape(temp_norm,size(temp_norm,1)*size(temp_norm,2),1);
yhat_res=reshape(yhat,size(yhat,1)*size(yhat,2),1);
%%
R = corrcoef(temp_norm_res,yhat_res);
r=R(2,1);
mse=immse(yhat,temp_norm);

