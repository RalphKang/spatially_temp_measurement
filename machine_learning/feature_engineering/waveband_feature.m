clc
clear all
% this code is used to collect statistical features from the entire waveband and
% characteristic bands
spect=load("train_data.txt");
%% check data
plot(1:6799,spect(12,:))
%% split spectrum
band(1,:,:)=spect(:,1:900); % water
band(2,:,:)=spect(:,3001:3900); %co
band(3,:,:)=spect(:,5101:6000);% co2
for i=1:3
    spectrum=squeeze(band(i,:,:));
    [feature_gain(i,:,:)]= para_reprst_fuc(spectrum);
end
%% whole features
[feature_whole]=para_reprst_fuc(spect);
%% reshape feature gain


feature_band1=squeeze(feature_gain(1,:,:));
feature_band2=squeeze(feature_gain(2,:,:));
feature_band3=squeeze(feature_gain(3,:,:));
% feature_whole=xlsread("feature_selection.xlsx");
feature_all=[feature_band1,feature_band2,feature_band3];
feature_whole_all=[feature_band1,feature_band2,feature_band3,feature_whole];
%% write matrix
writematrix(feature_all,"band_feat_nnorm_logspec.txt");
writematrix(feature_whole,"spec_feat_nnorm_logspec.txt");
writematrix(feature_whole_all,"spec_pls_band_nnorm_feat_logspec.txt");
%% write special feature
writematrix(feature_band2,"co_band_log.txt");
