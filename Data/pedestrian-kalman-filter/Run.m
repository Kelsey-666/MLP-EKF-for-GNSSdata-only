% Example script to load GNSS data and run Kalman filter

% Add user paths - change to your path
addpath(genpath("C:\Shared\PedestrianKalmanFilter"));

% Get a list of all data files - change to your path
files = dir("C:\Shared\PedestrianKalmanFilter\Data\*gnss.mat");

% Process all files
for i = 1:length(files)
    filename = fullfile(files(i).folder,files(i).name);

    % load data file
    load(filename,"gnss","rov_truth")

    % Run the Kalman filter
    result = PedestrianKalmanFilter(gnss);

    % Plot the position error compared to the truth
    num_epoch = size(result.x,1);
    [pos_err_n, ~, vel_err_n, ~] = ecef2nedError(...
        result.x(:,result.state_indx.pos), ...
        rov_truth.pos_E(:,1:num_epoch)',...
        result.x(:,result.state_indx.vel),...
        rov_truth.vel_E(:,1:num_epoch)');

    % Position error plot
    figure;
    plot(pos_err_n);
    legend("North","East","Down");
    xlabel("Time [s]");
    ylabel("Position error [m]");
    title(files(i).name,"Interpreter","none")
    ylim([-20 20]);

    % Velocity error plot
    if (0)
        figure;
        plot(vel_err_n);
        legend("North","East","Down");
        xlabel("Time [s]");
        ylabel("Velocity error [m]");
        title(files(i).name,"Interpreter","none")
    end

    drawnow
end
