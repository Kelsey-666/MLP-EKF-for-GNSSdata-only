% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.
 
classdef GnssMeasurements
    % Object to store GNSS measurements for the Kalman filter
    properties
        % Receiver local time
        tow (:, 1) double = []

        % GPS week
        week (:, 1) double = []

        % True receiver ECEF position
        pos_E (:, 3) double = []

        % True receiver ECEF velocity
        vel_E (:, 3) double = []

        % True receiver NED frame position
        pos_std_N (:, 3) double = [] 

        % True reference receiver position
        pos_ref_E (:, 3) double = [] 

        % epoch
        epoch (:, 1) double = []

        % pseudoranges corrected for ionosphere, troposphere and satellite 
        % biases [m]
        pr (:, 1) double = []

        % corrected phase measurements for ionosphere and troposphere and 
        % satellite biases [m]
        cr (:, 1) double = []

        % Doppler measurements [m/s]
        dr (:, 1) double = []

        % GNSS ID (0 = GPS, 2 = Galileo, 3 = BeiDou, 5 = QZSS)
        gnss_id (:, 1) double = []

        % Satellite ID
        sv_id (:, 1) double = []

        % u-blox signal ID (see protocol spec)
        sig_id (:, 1) double = []

        % u-blox access ID (see protocol spec)
        accs_id (:, 1) double = []

        % Not used
        iono (:, 1) double = []

        % PR measurement standard deviation [m]
        pr_noise (:, 1) double = []

        % carrier range measurement standard deviation [m]
        cr_noise (:, 1) double = []

        % Doppler measurement standard deviation [m/s]
        dr_noise (:, 1) double = []

        % satellite ECEF position
        sat_pos_E (:, 3) double = []

        % satellite ECEF velocity
        sat_vel_E (:, 3) double = []

        % satellite clock bias [m]
        sat_bias (:, 1) double = []

        % satellite clock drift [m/s]
        sat_drift (:, 1) double = []

        % Unique ID for each continuous period of phase lock
        track_idx (:, 1) double = []

        % Satellite elevation [radians?] 
        elev (:, 1) double = []

        % Satellite azimuth angle [radians?] todo: is this used?
        azim (:, 1) double = []

        % Signal wavelength [m]
        wavelength (:, 1) double = []

        % Signal to noise
        cno (:, 1) double = []

        % temporary
        xtId
        xtIdSv
    end

    methods (Access = public)
        function obj = GnssMeasurements()
        end

        function Validate(obj)
            % todo: we could check the array dimension here or check for 
            % missing fields        
        end
    end
end

