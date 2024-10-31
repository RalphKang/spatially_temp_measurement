% clc
% clear all
%the code is used to do all signal-representation feature extraction
function signal_representation()
%%
    data_folder="sample/"; % sample dir
    save_folder="exp2_20/";
    whole_dataset_length=24021 % train and validation
    for sample_index =0:whole_dataset_length
        str_index=num2str(sample_index);
        file_dir=strcat(data_folder,str_index,".csv");
        save_dir=strcat(save_folder,str_index,".csv");
        spectrum_org=load(file_dir);
        spectrum=log(spectrum_org);
        %% give the initial setting of groups 
        dim = size(spectrum);
        length=dim(1);
        sample_number=dim(2);
        order=3; % fitting order
        window_size=20;
        %%
        filename1=strcat('data_fit_new/','fitting','_','exp2','_',num2str(window_size),".txt");
        filename2=strcat('data_fit_new/','mse','_','exp2','_',num2str(window_size));
        %%
        spectrum_prcs=zeros(1,(int32(dim(1)/window_size)-1)*(order+1));
        % y_new=zeros(dim);
        x=[1:window_size];
        segments=int32(dim(1)/window_size)-1;
%         tic
            for i=1:segments
                i=i-1;
                y=spectrum((i*window_size)+1:(i+1)*window_size);
                f = fit(x.',y,'fourier1');
                p = coeffvalues(f);
                spectrum_prcs(1,i*(order+1)+1:(i+1)*(order+1))=p;        
            end
            writematrix(spectrum_prcs,save_dir)
%      toc
    end
end

