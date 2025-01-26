% Self-Adaptive LMS Without Desired Signal
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

%% Adaptive LMS Filtering
filter_order = 32;
mu = 0.005; % Step size
tic;
[error_signal, output_signal] = lms_filter(speech_noisy, speech_preprocessed, mu, filter_order);
execution_time = toc;
disp("Execution Time:");
disp(execution_time);

% Ensure real signals and normalize for playback
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
title('Filtered Speech Signal (LMS)');
xlabel('Time (s)');
ylabel('Amplitude');

%% Play Original and Filtered Signals
% disp('Playing original noisy speech...');
% sound(speech_noisy, Fs);
% pause(length(speech_noisy)/Fs + 1);
% disp('Playing filtered speech...');
% sound(output_signal, Fs);

% Ensure real signals and normalize for playback
speech_noisy = speech_noisy / max(abs(speech_noisy));
output_signal = real(output_signal) / max(abs(output_signal));

%% Calculate SNR
snr_value = calculate_snr(speech_preprocessed, error_signal);
disp(['SNR (dB): ', num2str(snr_value)]);

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

%% SNR Calculation Function
function snr_value = calculate_snr(clean_signal, noise_signal)
    power_clean = mean(clean_signal.^2); % Power of the clean signal
    power_noise = mean(noise_signal.^2); % Power of the noise signal
    snr_value = 10 * log10(power_clean / power_noise); 
end