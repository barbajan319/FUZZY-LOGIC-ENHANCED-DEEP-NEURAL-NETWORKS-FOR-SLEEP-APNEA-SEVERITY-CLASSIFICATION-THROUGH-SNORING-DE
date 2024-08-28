% Define parameters
f0_values = [100, 250, 500, 750]; % Frequencies in Hz
fs_values = [500, 2500]; % Sampling rates in samples/second
t = 0:0.1:0.1; % Time vector for 0.1 seconds

% Plot for each combination of f0 and fs
for i = 1:length(fs_values)
    fs = fs_values(i);
    for j = 1:length(f0_values)
        f0 = f0_values(j);
        n = 0:(1/fs):0.1; % Time vector for the given sampling rate
        x_n = cos(2*pi*f0*n); % Signal
        subplot(length(fs_values), length(f0_values), (i-1)*length(f0_values) + j);
        plot(n, x_n);
        title(sprintf('f0 = %d Hz, fs = %d samples/second', f0, fs));
        xlabel('Time (s)');
        ylabel('Amplitude');
    end
end
