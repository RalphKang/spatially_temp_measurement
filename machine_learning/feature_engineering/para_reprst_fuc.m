% function: this code is used to extract statistic features to represent
function[feature_gain]= para_reprst_fuc(spectrum)
%% feature extraction
mean_spc=mean(spectrum,2); % feature 1
std_spc=std(spectrum,0,2); % feature 2
min_spc=min(spectrum,[],2); % feature 3
max_spc=max(spectrum,[],2); % feature 4
quarti_1=quantile(spectrum,0.25,2); % feature 5
quarti_3=quantile(spectrum,0.75,2); % feature 6
median_spc=quantile(spectrum,0.5,2); % feature 7
skew_spc=skewness(spectrum,0,2); % feature 8
kurt_spc=kurtosis(spectrum,0,2); % feature 9
%%
ptp_spc=min_spc-max_spc;% peak to peak % feature 10
% spectrum_over_mean=spectrum-median_spc; %c persudo cross zero rate
ZCR=mean(abs(diff(sign(spectrum'))));
ZCR=ZCR'; % feature 11
rms_spc=rms(spectrum,2); % root of mean square; % feature 12
crest_factor=max_spc./rms_spc; % crest_factor; % % feature 13
dim2=size(spectrum,2);
spectrum_1=spectrum(:,1:dim2-1);% rms of spectrum speed
spectrum_2=spectrum(:,2:dim2);
spectrum_speed=spectrum_2-spectrum_1;
rms_sp_spc=rms(spectrum_speed,2);% feature 14
%% wavelet entropy
dim1=size(spectrum,1);
entp_shannon=zeros(dim1,1);
entp_energy=zeros(dim1,1);
for i=1:dim1
 entp_shannon(i)= wentropy(spectrum(i,:),'shannon'); %feature 20
 entp_energy(i)= wentropy(spectrum(i,:),'log energy'); %feature 21
end


%% Fourier Transform
fr_spec=fft(spectrum,dim2,2);
fr_spec_mag=abs(fr_spec(:,1:floor(dim2/2)+1));
pow_spec=1/dim2*fr_spec_mag.^2;
pow_spec(:,2:end-1)=pow_spec(:,2:end-1)*2; % power of spectrum
fr_phase=angle(fr_spec(:,1:floor(dim2/2)+1));
fr_mean_phase=mean(fr_phase,2); % mean phase of fft spectrum feature15
log_spc_ratio=log10(sum(pow_spec(:,1:floor(dim2/4)),2)./sum(pow_spec(:,floor(dim2/4):end),2)); %feature 16
spc_diff=sum(pow_spec(:,1:floor(dim2/4)),2)-sum(pow_spec(:,floor(dim2/4):end),2); %feature 17
mean_fft=mean(fr_spec_mag,2); % mean magnitude of fft feature 18
max_power=max(pow_spec,[],2); % max power of FFT feature 19
% plot(1:dim2,spectrum_inverse(1,:))
%% feature collection
feature_gain=[mean_spc,std_spc,min_spc,max_spc,quarti_1,quarti_3,median_spc,skew_spc,kurt_spc,...
    ptp_spc,ZCR,rms_spc,crest_factor,rms_sp_spc,fr_mean_phase,log_spc_ratio,spc_diff,mean_fft,max_power,entp_shannon,entp_energy];
max_feature=max(feature_gain,[],1);
min_feature=min(feature_gain,[],1);
norm_feature=(feature_gain-min_feature)./(max_feature-min_feature);
end

