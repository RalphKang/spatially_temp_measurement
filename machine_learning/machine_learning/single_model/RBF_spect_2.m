%% load data
function RBF_spect_2(dir1,dir2, feature_size)
spect=load(dir1);
temp=load(dir2);
spect_coef=spect(:,1:feature_size);
%% data normalization
min_spec=min(min(spect_coef,[],2));
max_spec=max(max(spect_coef,[],2));
norm_spec=(spect_coef-min_spec) ./(max_spec-min_spec);
%% split the dataset
rng(1); % set seed
% data_set=[spect_coef,temp];
data_set=[norm_spec,temp];
order=randperm(size(data_set, 1)); 
rdm_data=data_set(order,:);
train_set_amount=int32(0.85*size(data_set,1));
train_set=rdm_data(1:train_set_amount,:);
train_data=train_set(:,1:size(spect_coef,2))';
train_label=train_set(:,size(spect_coef,2)+1:end)';
test_set=rdm_data(train_set_amount+1:size(data_set,1),:);
test_data=test_set(:,1:size(spect_coef,2))';
test_label=test_set(:,size(spect_coef,2)+1:end)';

%% training RBF
mse_target=0.016:-0.002:0.001;
dim_mse_t=size(mse_target,2);
mse_train=ones(1,dim_mse_t);
mse_test=ones(1,dim_mse_t);
r2_train=zeros(1,dim_mse_t);
r2_test=zeros(1,dim_mse_t);
brk_count=0;
for ite=1:dim_mse_t
    % train model
    spread=1.0;
    MN=3000;
    net = newrb(train_data,train_label,mse_target(ite),spread,MN);
    % calculate performance of model
    train_hat=sim(net,train_data);
    [r2_train(ite),mse_train(ite)] = rsquare(train_label,train_hat);
    test_hat=sim(net,test_data);
    [r2_test(ite),mse_test(ite)] = rsquare(test_label,test_hat);
    out_put=[r2_train(ite),mse_train(ite),r2_test(ite),mse_test(ite)];
    % record model performance
    fid = fopen('result_record.xls','a');
    fprintf(fid,'%d \t ',out_put); 
    fprintf(fid,'\r\n');  % »»ÐÐ
    fclose(fid);
    % record best performance and model
    if mse_test(ite)== min(mse_test)
        writematrix(out_put,"perf_opt.xls")
        save("rbf_net.mat",'net')
        brk_count=0;
    end
    % early stop
    if mse_test(ite)~= min(mse_test)
        brk_count=brk_count+1;
    end
    if brk_count>5
        break
    end
    if mse_train(ite)-mse_target(ite)>0.002
        break
    end
end
%% common function
function [r,mse] = rsquare(y,f,varargin)
% Compute coefficient of determination of data fit model and RMSE
%
% [r2 rmse] = rsquare(y,f)
% [r2 rmse] = rsquare(y,f,c)
%
% RSQUARE computes the coefficient of determination (R-square) value from
% actual data Y and model data F. The code uses a general version of 
% R-square, based on comparing the variability of the estimation errors 
% with the variability of the original values. RSQUARE also outputs the
% root mean squared error (RMSE) for the user's convenience.
%
% Note: RSQUARE ignores comparisons involving NaN values.
% 
% INPUTS
%   Y       : Actual data
%   F       : Model fit
%
% OPTION
%   C       : Constant term in model
%             R-square may be a questionable measure of fit when no
%             constant term is included in the model.
%   [DEFAULT] TRUE : Use traditional R-square computation
%            FALSE : Uses alternate R-square computation for model
%                    without constant term [R2 = 1 - NORM(Y-F)/NORM(Y)]
%
% OUTPUT 
%   R2      : Coefficient of determination
%   RMSE    : Root mean squared error
%
% EXAMPLE
%   x = 0:0.1:10;
%   y = 2.*x + 1 + randn(size(x));
%   p = polyfit(x,y,1);
%   f = polyval(p,x);
%   [r2 rmse] = rsquare(y,f);
%   figure; plot(x,y,'b-');
%   hold on; plot(x,f,'r-');
%   title(strcat(['R2 = ' num2str(r2) '; RMSE = ' num2str(rmse)]))
%   
% Jered R Wells
% 11/17/11
% jered [dot] wells [at] duke [dot] edu
%
% v1.2 (02/14/2012)
%
% Thanks to John D'Errico for useful comments and insight which has helped
% to improve this code. His code POLYFITN was consulted in the inclusion of
% the C-option (REF. File ID: #34765).
if isempty(varargin); c = true; 
elseif length(varargin)>1; error 'Too many input arguments';
elseif ~islogical(varargin{1}); error 'C must be logical (TRUE||FALSE)'
else c = varargin{1}; 
end
% Compare inputs
if ~all(size(y)==size(f)); error 'Y and F must be the same size'; end
% Check for NaN
tmp = ~or(isnan(y),isnan(f));
y = y(tmp);
f = f(tmp);
if c; r2 = max(0,1 - sum((y(:)-f(:)).^2)/sum((y(:)-mean(y(:))).^2));
else r2 = 1 - sum((y(:)-f(:)).^2)/sum((y(:)).^2);
    if r2<0
    % http://web.maths.unsw.edu.au/~adelle/Garvan/Assays/GoodnessOfFit.html
        warning('Consider adding a constant term to your model') %#ok<WNTAG>
        r2 = 0;
    end
end
% rmse = sqrt(mean((y(:) - f(:)).^2));
mse=mean((y(:) - f(:)).^2);
r=sqrt(r2);
end
end
