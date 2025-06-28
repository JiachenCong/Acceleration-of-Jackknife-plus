rng(98765);
n_train=100;
n_test=100;
Round=50;
Lambda=[0.01,0.1,1];
P_number=[50,100,200];
alpha=0.1;
colNames={'Mean_Accuracy','Median_Accuracy','Standard Deviation_Accuracy','Mean_Time','Median_Time','Standard Deviation_Time'};
rowNames={'(n=100,p=50,λ=0.01)','(n=100,p=100,λ=0.01)','(n=100,p=200,λ=0.01)','(n=100,p=50,λ=0.1)','(n=100,p=100,λ=0.1)','(n=100,p=200,λ=0.1)','(n=100,p=50,λ=1)','(n=100,p=100,λ=1)','(n=100,p=200,λ=1)'};
result=zeros(9,6);
result_app=zeros(9,6);
result_mm=zeros(9,6);
result_orjkmm=zeros(9,6);
group=0;
for lambda=Lambda
for p=P_number
    group=group+1;
    total_time=zeros(Round,1);
    total_time_app=zeros(Round,1);
    total_time_mm=zeros(Round,1);
    total_time_orjkmm=zeros(Round,1);
    Accuracy=zeros(Round,1);
    Accuracy_app=zeros(Round,1);
    Accuracy_mm=zeros(Round,1);
    Accuracy_orjkmm=zeros(Round,1);
    for r=1:Round
        number_jk=0;
        number_jkapp=0;
        number_jkmm=0;
        number_orjkmm=0;
        beta = randn(p, 1);
        beta = beta / norm(beta) * sqrt(10);
        X=randn(n_train,p);
        Y_train = X * beta + randn(n_train, 1);
        X_test=randn(n_test,p);
        Y_test=X_test*beta+randn(n_test,1);
        I=eye(p);
        R_loo=zeros(n_train,1);%originial R_loo
        R_hat_loo=zeros(n_train,1);%R_loo constructed by approximate estimator
        pred_jk=zeros(n_test,2);%prediction intervals by original jackknife+
        pred_jkapp=zeros(n_test,2);%prediction intervals by fast jackknife+
        pred_jkmm=zeros(n_test,2);%prediction intervals by fast jackknife-minmax
        pred_orjkmm=zeros(n_test,2);%prediction intervals by original jackknife-minmax
        %Original Jackknife+
        tic;
        for i=1:n_train
            X_i = X([1:i-1, i+1:end], :);
            Y_i = Y_train([1:i-1, i+1:end]);
            I_i=eye(p);
            beta_i=(X_i' * X_i + lambda * I_i) \ (X_i' * Y_i);
            R_loo(i)=abs(Y_train(i)-X(i, :)*beta_i);
        end
        for j=1:n_test
            upper=zeros(n_train,1);
            lower=zeros(n_train,1);
            for i=1:n_train
                X_i = X([1:i-1, i+1:end], :);
                Y_i = Y_train([1:i-1, i+1:end]);
                I_i=eye(p);
                beta_i=(X_i' * X_i + lambda * I_i) \ (X_i' * Y_i);
                lower(i)=X_test(j,:)*beta_i-R_loo(i);
                upper(i)=X_test(j,:)*beta_i+R_loo(i);
            end
            pred_jk(j,1)=q_lower(lower,alpha);
            pred_jk(j,2)=q_upper(upper,alpha);
        end
        total_time(r) = toc;
        %jackknife+ with approximate leave-one-out estimators
        tic;
        J=X'*X+lambda*eye(p);
        beta_full = (X' * X + lambda * eye(p)) \ (X' * Y_train);
        for i=1:n_train
            x_i = X(i, :);
            r_i = -(Y_train(i) - x_i * beta_full);
            v = J \ x_i';
            beta_i_app = beta_full + v * r_i / (1 - x_i * v);
            R_hat_loo(i)=abs(Y_train(i)-X(i, :)*beta_i_app);
        end
        for j=1:n_test
            upper_app=zeros(n_train,1);
            lower_app=zeros(n_train,1);
            for i=1:n_train
                x_i = X(i, :);
                r_i = -(Y_train(i) - x_i * beta_full);
                v = J \ x_i';
                beta_i_app = beta_full + v * r_i / (1 - x_i * v);
                lower_app(i)=X_test(j,:)*beta_i_app-R_hat_loo(i);
                upper_app(i)=X_test(j,:)*beta_i_app+R_hat_loo(i);
            end
            pred_jkapp(j,1)=q_lower(lower_app,alpha);
            pred_jkapp(j,2)=q_upper(upper_app,alpha);
        end
        total_time_app(r)=toc;
        R_loo=zeros(n_train,1);
        %original jackknife-minmax
        tic;
        for i=1:n_train
            X_i = X([1:i-1, i+1:end], :);
            Y_i = Y_train([1:i-1, i+1:end]);
            I_i=eye(p);
            beta_i=(X_i' * X_i + lambda * I_i) \ (X_i' * Y_i);
            R_loo(i)=abs(Y_train(i)-X(i, :)*beta_i);
        end
        for j=1:n_test
            minmax=zeros(n_train,1);
            for i=1:n_train
                X_i = X([1:i-1, i+1:end], :);
                Y_i = Y_train([1:i-1, i+1:end]);
                I_i=eye(p);
                beta_i=(X_i' * X_i + lambda * I_i) \ (X_i' * Y_i);
                minmax(i)=X_test(j,:)*beta_i;
            end
            pred_orjkmm(j,1)=min(minmax)-q_upper(R_loo,alpha);
            pred_orjkmm(j,2)=max(minmax)+q_upper(R_loo,alpha);
        end
        total_time_orjkmm(r) = toc;
        R_hat_loo=zeros(n_train,1);
        %jackknife-minmax with approximate leave-one-out estimators
        tic;
        J=X'*X+lambda*eye(p);
        beta_full = (X' * X + lambda * eye(p)) \ (X' * Y_train);
        for i=1:n_train
            x_i = X(i, :);
            r_i = -(Y_train(i) - x_i * beta_full);
            v = J \ x_i';
            beta_i_app = beta_full + v * r_i / (1 - x_i * v);
            R_hat_loo(i)=abs(Y_train(i)-X(i, :)*beta_i_app);
        end
        for j=1:n_test
            minmax=zeros(n_train,1);
            for i=1:n_train
                x_i = X(i, :);
                r_i = -(Y_train(i) - x_i * beta_full);
                v = J \ x_i';
                beta_i_app = beta_full + v * r_i / (1 - x_i * v);
                minmax(i)=X_test(j,:)*beta_i_app;
            end
            pred_jkmm(j,1)=min(minmax)-q_upper(R_hat_loo,alpha);
            pred_jkmm(j,2)=max(minmax)+q_upper(R_hat_loo,alpha);
        end
        total_time_mm(r)=toc;
        %accuracy
        for j=1:n_test
            if (Y_test(j)>=pred_jk(j,1)) && (Y_test(j)<=pred_jk(j,2))
                number_jk=number_jk+1;
            end
            if (Y_test(j)>=pred_jkapp(j,1)) && (Y_test(j)<=pred_jkapp(j,2))
                number_jkapp=number_jkapp+1;
            end
            if (Y_test(j)>=pred_jkmm(j,1)) && (Y_test(j)<=pred_jkmm(j,2))
                number_jkmm=number_jkmm+1;
            end
            if (Y_test(j)>=pred_orjkmm(j,1)) && (Y_test(j)<=pred_orjkmm(j,2))
                number_orjkmm=number_orjkmm+1;
            end
        end
        Accuracy(r)=number_jk/100;
        Accuracy_app(r)=number_jkapp/100;
        Accuracy_mm(r)=number_jkmm/100;
        Accuracy_orjkmm(r)=number_orjkmm/100;
    end
    result(group,1)=mean(Accuracy);
    result(group,2)=median(Accuracy);
    result(group,3)=std(Accuracy);
    result(group,4)=mean(total_time);
    result(group,5)=median(total_time);
    result(group,6)=std(total_time);
    result_app(group,1)=mean(Accuracy_app);
    result_app(group,2)=median(Accuracy_app);
    result_app(group,3)=std(Accuracy_app);
    result_app(group,4)=mean(total_time_app);
    result_app(group,5)=median(total_time_app);
    result_app(group,6)=std(total_time_app);
    result_mm(group,1)=mean(Accuracy_mm);
    result_mm(group,2)=median(Accuracy_mm);
    result_mm(group,3)=std(Accuracy_mm);
    result_mm(group,4)=mean(total_time_mm);
    result_mm(group,5)=median(total_time_mm);
    result_mm(group,6)=std(total_time_mm);
    result_orjkmm(group,1)=mean(Accuracy_orjkmm);
    result_orjkmm(group,2)=median(Accuracy_orjkmm);
    result_orjkmm(group,3)=std(Accuracy_orjkmm);
    result_orjkmm(group,4)=mean(total_time_orjkmm);
    result_orjkmm(group,5)=median(total_time_orjkmm);
    result_orjkmm(group,6)=std(total_time_orjkmm);
end
end
disp('Result for Original Jackknife+')
T_1=array2table(result, 'RowNames', rowNames, 'VariableNames', colNames);
disp(T_1);
disp('Result for Jackknife+ by Approximate estimators')
T_2=array2table(result_app, 'RowNames', rowNames, 'VariableNames', colNames);
disp(T_2);
disp('Result for Jackknife-minmax by Approximate estimators')
T_3=array2table(result_mm, 'RowNames', rowNames, 'VariableNames', colNames);
disp(T_3);
disp('Result for Original Jackknife-minmax')
T_4=array2table(result_orjkmm, 'RowNames', rowNames, 'VariableNames', colNames);
disp(T_4);
writetable(T_1, 'JKplus_Results.xlsx', 'Sheet', 'Sheet1', 'WriteRowNames', true);
writetable(T_2, 'JKplus_Results_app.xlsx', 'Sheet', 'Sheet1', 'WriteRowNames', true);
writetable(T_3, 'JKplus_Results_mm.xlsx', 'Sheet', 'Sheet1', 'WriteRowNames', true);
writetable(T_4, 'JKplus_Results_orjkmm.xlsx', 'Sheet', 'Sheet1', 'WriteRowNames', true);
%quantile functions
function q_plus = q_upper(v, alpha)
    n = length(v);
    k = ceil((1 - alpha) * (n + 1));
    v_sorted = sort(v);
    k = min(max(k, 1), n); 
    q_plus = v_sorted(k);
end
function q_lower = q_lower(v, alpha)
    n = length(v);
    k = ceil(alpha * (n + 1));
    v_sorted = sort(v);
    k = min(max(k, 1), n);
    q_lower = v_sorted(k);
end
