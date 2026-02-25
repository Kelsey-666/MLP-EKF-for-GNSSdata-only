% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

classdef Constants
    % class to hold all constants
    properties (Constant)
        c = 299792458       % speed of light m/s
        GM = 3986004.418e8; % WGS84 gravitational constant m^3/s^2
        earth_rot_rate = 7.2921151467e-5; % Earth's rotation rate rad/s

        % angles
        rad2deg = (180/pi);
        deg2rad = (pi/180);

        % time
        secs_in_week = (24 * 7 * 60 * 60);
        beidou_week_offset = 1356; % Fixed week offset for Beidou
    end
end