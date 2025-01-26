% Self-Adaptive LMS with PSO for Parameter Optimization
clc; clear; close all;

%% Load Noisy Speech Signal
[speech_noisy, Fs] = audioread('speech_noisy.wav'); % Replace with your file
speech_noisy = speech_noisy(:, 1); % Single channel if stereo
t = (0:length(speech_noisy)-1)/Fs; % Time vector

%% Preprocessing: Spectral Subtraction
nfft = 1024;
window = hamming(256);
overlap = 128;

[S, F, T] = spectrogram(speech_noisy, window, overlap, nfft, Fs);
noise_est = mean(abs(S(:, 1:10)), 2); % Estimate noise from the first few frames
S_clean = abs(S) - noise_est; % Subtract noise spectrum
S_clean(S_clean < 0) = 0; % Avoid negative values
S_clean = S_clean .* exp(1j * angle(S)); % Reapply phase information

speech_preprocessed = istft(S_clean, Fs, 'Window', window, 'OverlapLength', overlap);
speech_preprocessed = real(speech_preprocessed); % Remove imaginary part

%% PSO Parameters
num_particles = 20;
max_iter = 50;
mu_min = 0.0001;
mu_max = 0.01;
filter_order_min = 8;
filter_order_max = 64;

% Initialize particles
params_particles = [mu_min + (mu_max - mu_min) * rand(num_particles, 1), ...
                    randi([filter_order_min, filter_order_max], num_particles, 1)];
velocities = zeros(size(params_particles));
personal_best = params_particles;
global_best = params_particles(1, :);

% Evaluate initial cost
personal_best_cost = arrayfun(@(i) pso_cost(params_particles(i, :), speech_noisy, speech_preprocessed), ...
                              1:num_particles);
[global_best_cost, idx] = min(personal_best_cost);
global_best = params_particles(idx, :);

% PSO main loop
for iter = 1:max_iter
    for i = 1:num_particles
        % Update velocity and position
        r1 = rand; r2 = rand;
        velocities(i, :) = 0.7 * velocities(i, :) + ...
                           1.5 * r1 * (personal_best(i, :) - params_particles(i, :)) + ...
                           1.5 * r2 * (global_best - params_particles(i, :));
        params_particles(i, :) = params_particles(i, :) + velocities(i, :);

        % Bound check
        params_particles(i, 1) = max(mu_min, min(params_particles(i, 1), mu_max)); % Bound \( \mu \)
        params_particles(i, 2) = max(filter_order_min, min(params_particles(i, 2), filter_order_max)); % Bound filter order

        % Evaluate new cost
        cost = pso_cost(params_particles(i, :), speech_noisy, speech_preprocessed);
        if cost < personal_best_cost(i)
            personal_best_cost(i) = cost;
            personal_best(i, :) = params_particles(i, :);
        end
    end

    % Update global best
    [global_best_cost, idx] = min(personal_best_cost);
    global_best = personal_best(idx, :);
end

% Optimized parameters
mu_optimized = global_best(1);
filter_order_optimized = round(global_best(2));

%% Adaptive LMS Filtering
[error_signal, output_signal] = lms_filter(speech_noisy, speech_preprocessed, mu_optimized, filter_order_optimized);

% Normalize signals for playback
speech_noisy = speech_noisy / max(abs(speech_noisy));
output_signal = real(output_signal) / max(abs(output_signal));

%% Plot Results
figure;
subplot(3, 1, 1);
plot(t, speech_noisy);
title('Noisy Speech Signal');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3, 1, 2);
plot(t, speech_preprocessed);
title('Preprocessed Speech Signal (Spectral Subtraction)');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3, 1, 3);
plot(t, output_signal);
title('Filtered Speech Signal (LMS with Optimized Parameters)');
xlabel('Time (s)');
ylabel('Amplitude');

%% Play Original and Filtered Signals
disp('Playing original noisy speech...');
sound(speech_noisy, Fs);
pause(length(speech_noisy)/Fs + 1);
disp('Playing filtered speech...');
sound(output_signal, Fs);

%% Define PSO Cost Function
function cost = pso_cost(params, noisy_signal, preprocessed_signal)
    mu = params(1);
    filter_order = round(params(2));
    [error_signal, ~] = lms_filter(noisy_signal, preprocessed_signal, mu, filter_order);
    cost = mean(error_signal.^2); % Mean squared error
end

%% Define LMS Filter Function
function [error_signal, output_signal] = lms_filter(noisy_signal, preprocessed_signal, mu, filter_order)
    N = length(noisy_signal);
    W = zeros(filter_order, 1); % Initialize filter coefficients
    output_signal = zeros(N, 1);
    error_signal = zeros(N, 1);

    for n = filter_order:N
        x = noisy_signal(n:-1:n-filter_order+1); % Current noisy signal segment
        output_signal(n) = W' * x; % LMS filter output
        error_signal(n) = preprocessed_signal(n) - output_signal(n); % Error signal
        W = W + 2 * mu * error_signal(n) * x; % Update filter coefficients
    end
end
