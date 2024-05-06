function [data_filter, Gbest_filter, Best_Fitness, tm] = FDS_MOMEDA_SA_PCS(data, Filter_Length, T, Tmax, plotMode, limit, coeff_sigma)
    
    x=data;
    t = zeros(length(x), 1);
    sigma = coeff_sigma*T;
    tm = zeros(length(x), round(T));

    for iter = 1:round(T)
        % ideal time-points of impulses
        for i = iter:T:length(x)
            % define normal distributions
            pulse = normpdf(1:length(x), i, sigma); 
            [~, maxIdx] = max(pulse);
            t = t + pulse';
        end

        % normalization
        t = t / max(t);
    
        % storage current t in tm
        tm(:, iter) = t;
    
        % reset t
        t = zeros(length(x), 1);
    end


    % Assign default values for inputs
    if (nargin < 7 || isempty(limit))
        limit = 1;  % Default limit is 1
    end

    % Validate the inputs
    if (nargin < 5 || isempty(Tmax))
        Tmax = 100;
    end
    if (nargin < 4 || isempty(T))
        error('SA_MOMEDA:InvalidInput', 'Input argument T must be provided.')
    end
    if (nargin < 2 || isempty(Filter_Length))
        Filter_Length = 100;
    end

    % Validate the inputs
    if (sum(size(data) > 1) > 1)
        error('SA_MOMEDA:InvalidInput', 'Input signal data must be a 1d vector.')
    elseif (sum(size(Tmax) > 1) ~= 0 || mod(Tmax, 1) ~= 0 || Tmax <= 0)
        error('SA_MOMEDA:InvalidInput', 'Input argument Tmax must be a positive integer scalar.')
    elseif (sum(size(Filter_Length) > 1) ~= 0 || Filter_Length <= 0 || mod(Filter_Length, 1) ~= 0)
        error('SA_MOMEDA:InvalidInput', 'Input argument Filter_Length must be a positive integer scalar.')
    elseif (Filter_Length > length(data))
        error('SA_MOMEDA:InvalidInput', 'The length of the filter must be less than or equal to the length of the data.')
    end

    % Force x into a column vector
    data = data(:);

    % Initial the parameters of SA algorithm
    initial_temperature = 1;
    temperature = initial_temperature;
    cooling_rate = 0.95;

    Angle_Num = Filter_Length - 1;

    % Define upper and lower bounds for each filter coefficient
    Pmax = limit; % Upper bounds
    Pmin = -limit; % Lower bounds

    % Initialize position
    Postion = Pmin + (Pmax - Pmin) .* rand(1, Angle_Num);

    % Initialize fitness
    fz = -KER_Fun_Pos(Postion, data, T, tm);

    % Initialize the personal experience
    Pbest_position = Postion; % personal best position
    Fitness = fz; % personal best fitness

    % Initialize the global best
    Gbest_position = Pbest_position;
    Gbest_Fitness = Fitness;

    % Iteration
    Best_Fitness = ones(Tmax, 1);
    time = ones(Tmax, 1);
    for stop = 1:Tmax
        time(stop) = stop;

        fprintf('Iteration %d\n', stop); % Display current iteration

        % Generate a random neighbor
        neighbor_position = Postion + 0.1 * randn(1, Angle_Num);

        % Calculate fitness for current and neighbor positions
        current_fitness = -KER_Fun_Pos(Postion, data, T, tm);
        neighbor_fitness = -KER_Fun_Pos(neighbor_position, data, T, tm);

        % Acceptance criterion based on fitness difference
        if neighbor_fitness < current_fitness || rand() < exp((current_fitness - neighbor_fitness) / temperature)
            Postion = neighbor_position;
        end

        % Cooling schedule
        temperature = temperature * cooling_rate;

        % Update fitness
        fz = -KER_Fun_Pos(Postion, data, T, tm);
 
        % Update personal best
        if fz <= Fitness
            Pbest_position = Postion;
            Fitness = fz;
        end

        % Update global best
        if Fitness < Gbest_Fitness
            Gbest_Fitness = Fitness;
            Gbest_position = Pbest_position;
        end

        Best_Fitness(stop) = Gbest_Fitness;
    end

    % Compute the filter (based on the angle parameters)
    Gbest_filter = 2 * Gbest_position - 1; % Scaling to [-1, 1]
    data_filter = filter(Gbest_filter, 1, data);
    Best_Fitness = -Best_Fitness;

    if plotMode == 1
        figure
        subplot(4, 1, 1);
        plot(data)
        title('Input Signal');
        ylabel('Value');
        xlabel('Sample Number');
        axis tight

        subplot(4, 1, 2);
        plot(data_filter)
        title('Signal Filtered');
        ylabel('Value');
        xlabel('Sample Number');
        axis tight

        subplot(4, 1, 3);
        stem(Gbest_filter)
        xlabel('Sample Number');
        ylabel('Value');
        title('Final Filter, Finite Impulse Response');
        axis tight

        subplot(4, 1, 4);
        plot(time, Best_Fitness)
        title('Fitness versus iteration');
        xlabel('Iteration');
        ylabel('Fitness');
        axis tight
    end
end

function Fitness = KER_Fun_Pos(theta, x, T, tm)
    S = size(theta);
    if S(2) == 1
        theta = theta';
    end

    % Compute the filter
    Gbest_filter = 2 * theta - 1; % Scaling
    x_filter = filter(Gbest_filter, 1, x);
    x_filter = x_filter - mean(x_filter);

    % Compute the norm of the filtered signal x_filter
    dnorm_x_f = multi_dnorm(x_filter, T, tm);
    Fitness = dnorm_x_f;
end

function result = multi_dnorm(x, T, tm)


    result = -1;
    for i = 1:round(T)
        ts = tm(:, i);
        mun = abs(ts)' * x / (sum(sum(x.^2))^(1 / 2));
        if mun > result
            result = mun;
        end
    end
end


