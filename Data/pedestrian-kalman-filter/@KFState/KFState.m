% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.
 
classdef KFState
    % Object that stores x and P for the Kalman filter and indexing
    % information
    properties
        % Current Kalman filter states
        x (:, 1) double = [];

        % Current Kalman filter covariance matrix
        P (:, :) double = [];

        % todo: make another class for this
        state_indx

        % total number of satellites seen in the data
        num_sv (1,1) double = 0;

        % total number of unique observations seen in the data
        num_obs (1,1) double = 0;

        % number of inter-GNSS biases
        inter_gnss (1,1) double = 0;

        % number of inter-band biases
        ibb (1,1) double = 0;

        % total number of states in the filter
        num_states
    end
end


