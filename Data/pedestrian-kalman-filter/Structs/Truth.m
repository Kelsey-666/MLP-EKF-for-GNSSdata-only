% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.
 
classdef Truth
    % a list of fields that are used to define the truth trajectory
    properties (SetAccess = public, GetAccess = public)
        pos_E       (:,:) double = []
        vel_E       (:,:) double = []
        acc_B       (:,:) double = []
        lat_lon_alt (:,:) double = []
        euler_N_B   (:,:) double = []
        vel_std_N   (:,:) double = []
        pos_std_N   (:,:) double = []
        distance    (1,:) double = []
        is_static   (1,:) logical = []
        % flag indicating true if we have extrapolated the trajectory
        % outside the valid range i.e. by assuming static
        extrapolated (1,:) logical = []
    end
end