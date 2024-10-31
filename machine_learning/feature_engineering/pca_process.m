clc
clear all
% the code is used to do PCA
data1=load("features.csv"); % load what kind of featrues are going to do so, such as the features after polynomial fitting
%% 
spectrum=data1;
mean_spec=mean(spectrum,1);
std_spc=std(spectrum,0,1);
spec_stand=(spectrum-mean_spec) ./std_spc;
%%
spec_stand_mean=mean(spec_stand,1);
%%
spec_max=max(spectrum,[],1);
spec_min=min(spectrum,[],1);
spec_norm=(spectrum-spec_min) ./(spec_max-spec_min);
%% pca process
[covarience,new_feature,latent,tsquare] = pca(spectrum);
precision=cumsum(latent)./sum(latent);
writematrix(new_feature ,"features_after_pca.csv")
%% stored for transforming test data
writematrix(mean_spec,'mean_feature.csv');
writematrix(covarience,'covarience_metrix.csv');
writematrix(std_spc,'std_spc.csv');

