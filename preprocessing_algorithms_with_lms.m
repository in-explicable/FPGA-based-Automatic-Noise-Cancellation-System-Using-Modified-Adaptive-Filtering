% Noise Reduction in Audio using Different Preprocessing Methods
clc;
clear;
close all;

% Step 1: Load Input Audio Signal
[file, path] = uigetfile('*.mp3', 'Select an Input Audio File'); % Prompt user to select a .wav file
audioPath = fullfile(path, file);
[x, Fs] = audioread(audioPath); % Read audio file
x = x(:,1); % Use single channel for simplicity
time = (0:length(x)-1) / Fs;

% Add noise for testing
rng(0); % For reproducibility
noise = 0.05 * randn(size(x)); % Add Gaussian noise
x_noisy = x + noise;
%x_noisy = x;

% Plot Original and Noisy Signals
figure;
subplot(3, 1, 1);
plot(time, x);
title('Original Signal');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3, 1, 2);
plot(time, noise);
title('Added Noise');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3, 1, 3);
plot(time, x_noisy);
title('Noisy Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Step 2: Noise Reduction Methods

% 1. Moving Average Filter
windowSize = 5; % Define the window size
x_ma = filter(ones(1, windowSize)/windowSize, 1, x_noisy);

% 2. Gaussian Smoothing
sigma = 2; % Standard deviation
windowSize_gauss = 15; % Ensure the window size covers multiple sigma
gaussKernel = fspecial('gaussian', [windowSize_gauss, 1], sigma);
x_gauss = conv(x_noisy, gaussKernel, 'same');

% 3. Savitzky-Golay Filter
polyOrder = 3; % Polynomial order
frameSize = 31; % Frame size (must be odd and > polyOrder)
x_sg = sgolayfilt(x_noisy, polyOrder, frameSize);

% 4. Wavelet Transform
% wavelet = 'db4'; % Daubechies wavelet with 4 vanishing moments
% level = 5; % Decomposition level
% [c, l] = wavedec(x_noisy, level, wavelet); % Wavelet decomposition
% sigma_noise = median(abs(c(1:l(1)))) / 0.6745; % Estimate noise level
% threshold = sigma_noise * sqrt(2 * log(length(x_noisy))); % Universal threshold
% c_thresh = wthresh(c, 's', threshold); % Apply soft thresholding
% x_wavelet = waverec(c_thresh, l, wavelet); % Reconstruct signal
% Parameters for Spectral Subtraction
frameLength = 256; % Frame length
overlap = frameLength / 2; % 50% overlap
win = hamming(frameLength); % Window function

% Perform STFT
[stft_noisy, f, t] = stft(x_noisy, Fs, 'Window', win, 'OverlapLength', overlap, 'FFTLength', frameLength);

% Estimate noise spectrum from initial frames
noiseFrames = 10; % Number of initial frames used for noise estimation
noise_spectrum = mean(abs(stft_noisy(:, 1:noiseFrames)).^2, 2);

% Spectral Subtraction
stft_clean = abs(stft_noisy).^2 - noise_spectrum; % Subtract noise power spectrum
stft_clean = max(stft_clean, 0); % Ensure no negative values
stft_clean = sqrt(stft_clean) .* exp(1j * angle(stft_noisy)); % Restore phase

% Inverse STFT
x_wavelet = istft(stft_clean, Fs, 'Window', win, 'OverlapLength', overlap, 'FFTLength', frameLength);
x_wavelet = real(x_wavelet);

if length(x_wavelet) < length(x_noisy)
    x_wavelet = [x_wavelet; zeros(length(x_noisy) - length(x_wavelet), 1)];
end

% Step 3: Analyze Results
figure;
methods = {'Moving Average', 'Gaussian Smoothing', 'Savitzky-Golay', 'Spectral Subtraction'};
processed_signals = {x_ma, x_gauss, x_sg, x_wavelet};

% Crop or zero-pad processed signals to match the length of the original signal
for i = 1:length(processed_signals)
    len_diff = length(processed_signals{i}) - length(x);
    if len_diff > 0
        % Crop if longer
        processed_signals{i} = processed_signals{i}(1:length(x));
    elseif len_diff < 0
        % Zero-pad if shorter
        processed_signals{i} = [processed_signals{i}; zeros(-len_diff, 1)];
    end
end

% Update the time vector (based on the length of the original signal)
time = (0:length(x)-1) / Fs;

% Plot the results
figure;
for i = 1:length(processed_signals)
    subplot(length(processed_signals), 1, i);
    plot(time, processed_signals{i});
    title(['Denoised Signal: ' methods{i}]);
    xlabel('Time (s)');
    ylabel('Amplitude');
end


% % Step 4: Save Results for Comparison
% audiowrite('denoised_MA.wav', x_ma, Fs);
% audiowrite('denoised_Gaussian.wav', x_gauss, Fs);
% audiowrite('denoised_SG.wav', x_sg, Fs);
% audiowrite('denoised_Wavelet.wav', x_wavelet, Fs);

% Listen to the results (optional)
% disp('Playing Original Noisy Signal...');
% sound(x_noisy, Fs);
% pause(length(x_noisy)/Fs + 2); % Wait for playback to finish
% 
% for i = 1:5
%     disp(['Playing ' methods{i} ' Filtered Signal...']);
%     sound(processed_signals{i}, Fs);
%     pause(length(x_noisy)/Fs + 2);
% end

disp('All signals processed and saved.');

% Extend previous code with LMS implementation and analysis
% Assuming the variables `x`, `x_noisy`, and `Fs` from the previous code are already loaded

% Step 5: LMS Adaptive Filtering
% LMS parameters
% Adaptive LMS Filtering
filterOrder = 32; % Set filter order
mu = 0.01; % Step size
desired = x;
signal = x_noisy;
numIterations = length(signal); % Length of the signal
N = length(x_noisy); 

% Initialize variables
weights = zeros(filterOrder, 1); % Initial weights
x_lms = zeros(size(signal)); % Filtered signal
error = zeros(size(signal)); % Error signal

% Adaptive filtering loop
for n = filterOrder:numIterations
    % Extract input vector
    inputVector = signal(n:-1:n-filterOrder+1);

    % Calculate output
    x_lms(n) = weights' * inputVector;

    % Calculate error
    error(n) = desired(n) - x_lms(n);

    % Update weights
    weights = weights + 2 * mu * error(n) * inputVector;
end

% Step 6: LMS with Preprocessed Inputs
% Initialize outputs
lms_outputs = cell(1, 4);
preprocessed_signals = {x_ma, x_gauss, x_sg, x_wavelet};

for i = 1:4
    signal = preprocessed_signals{i}; % Preprocessed signal
    weights = zeros(filterOrder, 1); % Reset weights
    
    % LMS loop for preprocessed signal
    for n = filterOrder:length(signal)
        inputVector = signal(n:-1:n-filterOrder+1);
        lms_outputs{i}(n) = weights' * inputVector; % LMS output
        error(n) = desired(n) - lms_outputs{i}(n); % Error signal
        weights = weights + 2 * mu * error(n) * inputVector; % Update weights
    end
end

% Step 7: Analyze Results
figure;
subplot(3, 1, 1);
plot(time, x, 'g');
title('Original Signal');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3, 1, 2);
plot(time, x_noisy, 'r');
title('Noisy Signal');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3, 1, 3);
plot(time, x_lms, 'b');
title('LMS Filtered Signal (Without Preprocessing)');
xlabel('Time (s)');
ylabel('Amplitude');

figure;
for i = 1:4
    subplot(4, 1, i);
    plot(time, lms_outputs{i}, 'b');
    title(['LMS Filtered Signal with Preprocessing: ' methods{i}]);
    xlabel('Time (s)');
    ylabel('Amplitude');
end

% Step 8: Save LMS Results for Comparison
% audiowrite('lms_filtered_no_preprocessing.wav', x_lms, Fs);
% 
% for i = 1:5
%     audiowrite(['lms_filtered_with_' methods{i} '.wav'], lms_outputs{i}, Fs);
% end

% Step 9: Evaluate Performance Metrics
fprintf('\nPerformance Analysis:\n');
snr_original = snr(x, x_noisy - x);
fprintf('SNR of Noisy Signal: %.2f dB\n', snr_original);

snr_lms_no_preprocessing = snr(x, x - x_lms);
fprintf('SNR after LMS (No Preprocessing): %.2f dB\n', snr_lms_no_preprocessing);

for i = 1:4
    snr_lms_preprocessed = snr(x, x - lms_outputs{i}');
    fprintf('SNR after LMS with %s: %.2f dB\n', methods{i}, snr_lms_preprocessed);
end

% Extend previous code with RLS and NLMS implementations

% RLS Parameters
lambda = 0.99; % Forgetting factor
delta = 0.001; % Small positive constant for initialization

% Preallocate RLS output
x_rls = zeros(N, 1);
P = eye(filterOrder) / delta; % Initialize inverse correlation matrix
weights_rls = zeros(filterOrder, 1);

% RLS Adaptive Filtering Loop
for n = filterOrder:N
    inputVector = x_noisy(n:-1:n-filterOrder+1); % Input vector
    k = (P * inputVector) / (lambda + inputVector' * P * inputVector); % Gain vector
    x_rls(n) = weights_rls' * inputVector; % RLS output
    error_rls = desired(n) - x_rls(n); % Error signal
    weights_rls = weights_rls + k * error_rls; % Update weights
    P = (P - k * inputVector' * P) / lambda; % Update inverse correlation matrix
end

% RLS with Preprocessed Inputs
rls_outputs = cell(1, 4);
for i = 1:4
    signal = preprocessed_signals{i};
    P = eye(filterOrder) / delta; % Reset P
    weights_rls = zeros(filterOrder, 1); % Reset weights
    for n = filterOrder:N
        inputVector = signal(n:-1:n-filterOrder+1);
        k = (P * inputVector) / (lambda + inputVector' * P * inputVector); % Gain vector
        rls_outputs{i}(n) = weights_rls' * inputVector; % RLS output
        error_rls = desired(n) - rls_outputs{i}(n); % Error signal
        weights_rls = weights_rls + k * error_rls; % Update weights
        P = (P - k * inputVector' * P) / lambda; % Update P
    end
end

% NLMS Parameters
epsilon = 1e-6; % Small constant to avoid division by zero
x_nlms = zeros(N, 1); % Preallocate NLMS output
weights_nlms = zeros(filterOrder, 1); % Initialize weights

% NLMS Adaptive Filtering Loop
for n = filterOrder:N
    inputVector = x_noisy(n:-1:n-filterOrder+1); % Input vector
    norm_factor = epsilon + inputVector' * inputVector; % Normalization factor
    x_nlms(n) = weights_nlms' * inputVector; % NLMS output
    error_nlms = desired(n) - x_nlms(n); % Error signal
    weights_nlms = weights_nlms + 2 * mu * error_nlms * inputVector / norm_factor; % Update weights
end

% NLMS with Preprocessed Inputs
nlms_outputs = cell(1, 4);
for i = 1:4
    signal = preprocessed_signals{i};
    weights_nlms = zeros(filterOrder, 1); % Reset weights
    for n = filterOrder:N
        inputVector = signal(n:-1:n-filterOrder+1);
        norm_factor = epsilon + inputVector' * inputVector; % Normalization factor
        nlms_outputs{i}(n) = weights_nlms' * inputVector; % NLMS output
        error_nlms = desired(n) - nlms_outputs{i}(n); % Error signal
        weights_nlms = weights_nlms + 2 * mu * error_nlms * inputVector / norm_factor; % Update weights
    end
end

% Analyze Results for RLS and NLMS
figure;
methods_adaptive = {'LMS', 'RLS', 'NLMS'};
adaptive_signals = {x_lms, x_rls, x_nlms};
for i = 1:3
    subplot(3, 1, i);
    plot(time, adaptive_signals{i});
    title(['Adaptive Filter Output: ' methods_adaptive{i}]);
    xlabel('Time (s)');
    ylabel('Amplitude');
end

figure;
for i = 1:4
    subplot(4, 1, i);
    plot(time, rls_outputs{i}, 'r');
    hold on;
    plot(time, nlms_outputs{i}, 'b');
    hold off;
    title(['RLS (Red) and NLMS (Blue) with Preprocessing: ' methods{i}]);
    xlabel('Time (s)');
    ylabel('Amplitude');
end

% Save RLS and NLMS Results
audiowrite('rls_filtered_no_preprocessing.wav', x_rls, Fs);
audiowrite('nlms_filtered_no_preprocessing.wav', x_nlms, Fs);

for i = 1:4
    audiowrite(['rls_filtered_with_' methods{i} '.wav'], rls_outputs{i}, Fs);
    audiowrite(['nlms_filtered_with_' methods{i} '.wav'], nlms_outputs{i}, Fs);
end

% Evaluate Performance Metrics for RLS and NLMS
fprintf('\nAdaptive Filter Performance Analysis:\n');
snr_rls_no_preprocessing = snr(x, x - x_rls);
snr_nlms_no_preprocessing = snr(x, x - x_nlms);
fprintf('SNR after RLS (No Preprocessing): %.2f dB\n', snr_rls_no_preprocessing);
fprintf('SNR after NLMS (No Preprocessing): %.2f dB\n', snr_nlms_no_preprocessing);

for i = 1:4
    snr_rls_preprocessed = snr(x, x - rls_outputs{i}');
    snr_nlms_preprocessed = snr(x, x - nlms_outputs{i}');
    fprintf('SNR after RLS with %s: %.2f dB\n', methods{i}, snr_rls_preprocessed);
    fprintf('SNR after NLMS with %s: %.2f dB\n', methods{i}, snr_nlms_preprocessed);
end

% Parameters for PSO
numParticles = 30;    % Number of particles
numIterations = 50;   % Number of iterations
dim = 2;              % Dimensions (learning rate and forgetting factor)
bounds = [0.001, 0.1; % Lower and upper bounds for mu (LMS/NLMS)
          0.9, 1.0];  % Lower and upper bounds for lambda (RLS)
c1 = 1.5;             % Cognitive coefficient
c2 = 1.5;             % Social coefficient
w = 0.8;              % Inertia weight

% Initialization
particles = rand(numParticles, dim) .* (bounds(:,2) - bounds(:,1))' + bounds(:,1)';
velocities = zeros(numParticles, dim);
pbest = particles;    % Personal best positions
pbest_scores = inf(numParticles, 1); % Personal best scores
gbest = particles(1, :); % Global best position
gbest_score = inf;

% Objective Function: SNR Maximization
objectiveFunction = @(params) -evaluateSNR(params, x, x_noisy, filterOrder, preprocessed_signals);

% PSO Main Loop
for iter = 1:numIterations
    for i = 1:numParticles
        % Evaluate fitness
        fitness = objectiveFunction(particles(i, :));
        
        % Update personal best
        if fitness < pbest_scores(i)
            pbest_scores(i) = fitness;
            pbest(i, :) = particles(i, :);
        end
        
        % Update global best
        if fitness < gbest_score
            gbest_score = fitness;
            gbest = particles(i, :);
        end
    end
    
    % Update particle velocities and positions
    for i = 1:numParticles
        r1 = rand(1, dim);
        r2 = rand(1, dim);
        velocities(i, :) = w * velocities(i, :) ...
                         + c1 * r1 .* (pbest(i, :) - particles(i, :)) ...
                         + c2 * r2 .* (gbest - particles(i, :));
                     
        particles(i, :) = particles(i, :) + velocities(i, :);
        
        % Enforce bounds
        particles(i, :) = max(particles(i, :), bounds(:,1)');
        particles(i, :) = min(particles(i, :), bounds(:,2)');
    end
end

% Optimized Parameters
mu_opt = gbest(1);
lambda_opt = gbest(2);

fprintf('Optimized Learning Rate (mu): %.4f\n', mu_opt);
fprintf('Optimized Forgetting Factor (lambda): %.4f\n', lambda_opt);

% Use Optimized Parameters in LMS, RLS, and NLMS
% LMS
weights = zeros(filterOrder, 1);
x_lms_opt = zeros(N, 1);
for n = filterOrder:N
    inputVector = x_noisy(n:-1:n-filterOrder+1);
    x_lms_opt(n) = weights' * inputVector;
    error = desired(n) - x_lms_opt(n);
    weights = weights + 2 * mu_opt * error * inputVector;
end

% RLS
P = eye(filterOrder) / delta;
weights_rls = zeros(filterOrder, 1);
x_rls_opt = zeros(N, 1);
for n = filterOrder:N
    inputVector = x_noisy(n:-1:n-filterOrder+1);
    k = (P * inputVector) / (lambda_opt + inputVector' * P * inputVector);
    x_rls_opt(n) = weights_rls' * inputVector;
    error_rls = desired(n) - x_rls_opt(n);
    weights_rls = weights_rls + k * error_rls;
    P = (P - k * inputVector' * P) / lambda_opt;
end

% NLMS
weights_nlms = zeros(filterOrder, 1);
x_nlms_opt = zeros(N, 1);
for n = filterOrder:N
    inputVector = x_noisy(n:-1:n-filterOrder+1);
    norm_factor = epsilon + inputVector' * inputVector;
    x_nlms_opt(n) = weights_nlms' * inputVector;
    error_nlms = desired(n) - x_nlms_opt(n);
    weights_nlms = weights_nlms + 2 * mu_opt * error_nlms * inputVector / norm_factor;
end

% Save and Analyze Results
figure;
subplot(3, 1, 1);
plot(time, x_lms_opt);
title('Optimized LMS Output');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3, 1, 2);
plot(time, x_rls_opt);
title('Optimized RLS Output');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3, 1, 3);
plot(time, x_nlms_opt);
title('Optimized NLMS Output');
xlabel('Time (s)');
ylabel('Amplitude');

% Save Outputs
audiowrite('optimized_lms.wav', x_lms_opt, Fs);
audiowrite('optimized_rls.wav', x_rls_opt, Fs);
audiowrite('optimized_nlms.wav', x_nlms_opt, Fs);

% Evaluate Optimized SNR
snr_lms_opt = snr(x, x - x_lms_opt);
snr_rls_opt = snr(x, x - x_rls_opt);
snr_nlms_opt = snr(x, x - x_nlms_opt);

fprintf('SNR after Optimized LMS: %.2f dB\n', snr_lms_opt);
fprintf('SNR after Optimized RLS: %.2f dB\n', snr_rls_opt);
fprintf('SNR after Optimized NLMS: %.2f dB\n', snr_nlms_opt);

% Objective Function Definition
function snr_value = evaluateSNR(params, clean, noisy, filterOrder, preprocessed_signals)
    mu = params(1);
    lambda = params(2);
    N = length(noisy);
    % LMS
    weights = zeros(filterOrder, 1);
    output = zeros(N, 1);
    for n = filterOrder:N
        inputVector = noisy(n:-1:n-filterOrder+1);
        output(n) = weights' * inputVector;
        error = clean(n) - output(n);
        weights = weights + 2 * mu * error * inputVector;
    end
    snr_value = snr(clean, clean - output);
end
