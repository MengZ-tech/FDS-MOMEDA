load signals_simulated.mat

figure;
subplot(5, 1, 1);
plot(signal1);

subplot(5, 1, 2);
plot(signal2);

subplot(5, 1, 3);
plot(signal3);

subplot(5, 1, 4);
plot(signal4);

subplot(5, 1, 5); %sum of signals
plot(signal5);

[data_filter,Gbest_filter,Best_Fitness]=FDS_MOMEDA_SA_PCS(signal5, 200, 100, 2000, 1, 1, 0.01);