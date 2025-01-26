%% ------------------------------------------------------------------------
% Master Thesis -
% Implementation of adaptive filtering algorithms for noise
% cancellation
% Author: Tanja Lampl
% Task: Adaptive noise cancellation without reference signal
% Last change: 2020-04-22
%% ------------------------------------------------------------------------
clear;
close all;
% The goal of the adaptive noise cancellation is to estimate the desired
% signal d(n) from a noise-corrupted observation x(n) = d(n) + v1(n)
%% Adaptive filter
N = 10000; % number of samples
R = 100; % ensemble average over 100 runs
nord = 12; % filter order
% LMS coefficients
mu = 0.002; % step size
% NLMS coefficients
beta = 0.25; % normalized step size
% RLS coefficients
delta = 0.001;
lambda = 1; % exponential weighting factor
% create arrays of all zeros
MSE_LMS = zeros(N,1);
MSE_NLMS = zeros(N,1);
LSE_RLS = zeros(N,1);
dhat_LMS = zeros(N,1);
dhat_NLMS = zeros(N,1);
dhat_RLS = zeros(N,1);
err_LMS = zeros(N,1);
err_NLMS = zeros(N,1);
err_RLS = zeros(N,1);
%for r=1:R % used for computation of learning curves
 %% Noise-corrupted observation: x(n) = d(n) + v1(n)
 d = sin((1:N)*0.05*pi); % desired signal
g = randn(1,N)*0.25; % Gaussian white noise with a variance of 0.25
v1= filter(1,[1 -0.8],g); % filtered white noise
x = d + v1; % noisy process

 % plot of d(n) and x(n)
 figure(1)
 subplot(2,1,1)
 plot(d(1:1000),':k')
 hold on

 plot(x(1:1000),'k')
 legend("d(n)", "x(n)")
 title("Noise-corrupted observation");
 xlabel('samples n')
 ylabel('amplitude')
 axis([0 1000 -3 3])
 grid on

 %% Reference signal
 % Here, the reference signal v2(n) is not known.
 % In that case, it is possible to derive a reference signal by
 % delaying the noisy process x(n) = d(n) + v1(n).
 % The delayed signal x(n-n0) is used as the reference signal for
 % the canceller.
 n0 = 25; % delay of 25 samples
 len = N - n0; % reduced vector length
 x_del = zeros(N,1); % create array of all zeros

 % generate delayed signal
 for i = 1:len
 x_del(i) = x(i+n0);
 end
 % plot of x_del(n)
 figure(2)
 subplot(2,1,1)
 plot(x(1:1000),':k')
hold on
plot(x_del(1:1000),'k')
legend("x(n)", "x(n-n0)")
title("Reference signal x(n-n0)");
xlabel('samples n')
ylabel('amplitude')
axis([0 1000 -3 3])
grid on
% create arrays of all zeros
W_LMS = zeros(nord,1);
W_NLMS = zeros(nord,1);
W_RLS = zeros(nord,1);
U = zeros(nord,1);
P = ((1/delta)*eye(nord,nord));
for i=1:N
U = [x_del(i)
U(1:(nord-1))];
x_n = x(i);
%% LMS Algorithm
% Step 1: Filtering
y_LMS = (W_LMS'*U);
dhat_LMS(i) = (dhat_LMS(i)+y_LMS); 

% Step 2: Error Estimation
E_LMS = (x_n-y_LMS);
err_LMS(i) = err_LMS(i)+E_LMS;
% Step 3: Tap-weight vector adaptation
W_LMS = (W_LMS+(mu*E_LMS*U));
%% NLMS Algorithm
% Step 1: Filtering
y_NLMS = (W_NLMS'*U);
dhat_NLMS(i) = (dhat_NLMS(i)+y_NLMS);
% Step 2: Error Estimation
E_NLMS = (x_n-y_NLMS);
err_NLMS(i) = err_NLMS(i)+E_NLMS;
% Step 3: Tap-weight vector adaptation
W_NLMS = (W_NLMS+((beta/((norm(U)^2)))*conj(E_NLMS)*U));
%% RLS Algorithm
% Step 1: Computing the gain vector
g = (((1/lambda)*P*U)/(1+((1/lambda)*U'*P*U)));
% Step 2: Filtering
y_RLS = (W_RLS'*U);
dhat_RLS(i) = (dhat_RLS(i)+y_RLS);
% Step 3: Error Estimation
E_RLS = (x_n-y_RLS);
err_RLS(i) = err_RLS(i)+E_RLS;
% Step 4: Tap-weight vector adaptation
W_RLS = W_RLS+g*conj(E_RLS);
%Step 5: Correlation Update
P = (((1/(lambda))*P)-((1/lambda)*g*U'*P));
%% Error performance
MSE_LMS(i) = norm(MSE_LMS(i)+(abs(E_LMS)^2));
MSE_NLMS(i) = norm(MSE_NLMS(i)+(abs(E_NLMS)^2));
LSE_RLS(i) = norm(LSE_RLS(i)+(abs(E_RLS)^2));
end
%end
%% Error performance
MSE_LMS = MSE_LMS/R;
MSE_NLMS = MSE_NLMS/R;
LSE_RLS = LSE_RLS/R;
% plot estimate of d(n)
figure(3)
plot(dhat_LMS(1:1000),'b');
title("LMS - Estimate of d(n)");
xlabel('samples n')
ylabel('amplitude')
axis([0 1000 -3 3])
grid on
figure(4)
plot(dhat_NLMS(1:1000),'b');
title("NLMS - Estimate of d(n)");

xlabel('samples n ')
ylabel('amplitude')
axis([0 1000 -3 3])
grid on
figure(5)
plot(dhat_RLS(1:1000),'b');
title("RLS - Estimate of d(n)");
xlabel('samples n')
ylabel('amplitude')
axis([0 1000 -3 3])
grid on
%% Result of adaptive noise cancellation
figure(6)
plot(dhat_LMS,':k');
hold on
plot(dhat_NLMS,'k');
hold on
plot(dhat_RLS,'--k');
legend("dhat(n) LMS", "dhat(n) NMLS","dhat(n) RLS");
hold off
title("Result of adaptive noise cancellation")
xlabel('samples n')
ylabel('amplitude')
axis([0 1000 -3 3]);
grid on;
%% Plot learning curves
figure(10)
plot(MSE_LMS(1:1000),':k')
ylabel('ensemble-average squared error')
xlabel('number of iterations')
title('LMS - Convergence rate ')
axis([0 1000 0 1])
grid on
figure(11)
plot(MSE_NLMS(1:1000),':k')
ylabel('ensemble-average squared error')
xlabel('number of iterations')
title('NLMS- Convergence rate ')
axis([0 1000 0 1])
grid on
figure(12)
plot(LSE_RLS(1:1000),':k')
ylabel('ensemble-average squared error')
xlabel('number of iterations')
title('RLS - Convergence rate ')
axis([0 1000 0 1])
grid on

