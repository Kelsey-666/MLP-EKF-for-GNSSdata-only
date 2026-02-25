% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

function gnss = ConvertToGnss(rov_proc, rov_solver_proc, rov_truth)
% Change input data into the format needed for the Kalman filter
%
% Args:
%   rov_proc:        proc class for rover
%   rov_solver_proc: solver_proc class for rover

nrow = size(rov_proc.signal.internal.pr,1);
ncol = size(rov_proc.signal.internal.pr,2);

% Calculates the predicted geometric range so we remove this from the 
% predicted value.
[rov_pred_pr, rov_pred_dr] = ...
    CalcPredictedObs(rov_proc.satellite, rov_proc.signal, 0);

epoch = repmat(1:length(rov_proc.receiver.rover_time_est),nrow,1);
sv_idx = rov_proc.signal.internal.sv_idx;

pr = rov_proc.signal.internal.pr;
dr = rov_proc.signal.internal.doppler .* rov_proc.signal.internal.lambda;
% replace Dopplers with delta ranges if they are available
dr_ok = ~isnan(rov_proc.signal.internal.delta_range);
if (~isempty(rov_proc.signal.internal.delta_range) && false)
    delta_range = rov_proc.signal.internal.delta_range .* rov_proc.signal.internal.lambda;
    dr(dr_ok) = delta_range(dr_ok);
end

ms_in_m = Constants.c*0.001;
clock_bias_round_1ms = round(rov_proc.receiver.clock_bias/ms_in_m)*ms_in_m;
geom_range_rov = rov_proc.satellite.internal.linearised.pr(sv_idx, :) - ...
    clock_bias_round_1ms;
geom_range_rate_rov = rov_proc.satellite.internal.linearised.doppler(sv_idx, :);

% compensate the range for the geometric range and clock bias as these
% are applied in SimKalman
pr_corr = rov_pred_pr - geom_range_rov;
dr_corr = rov_pred_dr - geom_range_rate_rov;

gnss_id = repmat(rov_proc.satellite.internal.gnss_id(sv_idx),1,ncol);
sv_id = repmat(rov_proc.satellite.internal.sv_id(sv_idx),1,ncol);

% satellite position, velocity and clock
sat_pos_x = rov_proc.satellite.internal.pos_E(sv_idx,:,1);
sat_pos_y = rov_proc.satellite.internal.pos_E(sv_idx,:,2);
sat_pos_z = rov_proc.satellite.internal.pos_E(sv_idx,:,3);
sat_vel_x = rov_proc.satellite.internal.vel_E(sv_idx,:,1);
sat_vel_y = rov_proc.satellite.internal.vel_E(sv_idx,:,2);
sat_vel_z = rov_proc.satellite.internal.vel_E(sv_idx,:,3);
sat_bias  = rov_proc.satellite.internal.clk(sv_idx,:);
sat_drift = rov_proc.satellite.internal.clk_drift(sv_idx,:);

% use a simple noise model
prNoise = rov_solver_proc.signal.pr_noise;
drNoise = rov_solver_proc.signal.do_noise;

wavelength = repmat(rov_proc.signal.internal.lambda,1,size(rov_proc.signal.internal.pr,2));
cno = rov_proc.signal.internal.cno;

%% set criteria for using measurements
elevMask = 10 * pi/180;
ok = (~isnan(pr) & ~isnan(pr_corr)) & ...
    ~isnan(rov_proc.satellite.internal.sat_elev_N(sv_idx,:)) & ...
    ~isnan(rov_proc.satellite.internal.clk(sv_idx,:)) & ...
    (rov_proc.satellite.internal.sat_elev_N(sv_idx,:) > elevMask);

if (sum(ok,"all") == 0)
    error("No valid measurements found");
end

% store per epoch information - see documentation of fields in
% GnssMeasurements class
gnss = GnssMeasurements;
gnss.tow = rov_proc.receiver.rover_time_est'; % estimated time (corresponds to time compensated by the clock bias)
gnss.week = rov_proc.receiver.week'; % estimated time (corresponds to time compensated by the clock bias)
gnss.pos_E = rov_truth.pos_E';
gnss.vel_E = rov_truth.vel_E';
gnss.pos_std_N = rov_truth.pos_std_N';

% store per measurement information
gnss.epoch = epoch(ok);
gnss.pr = pr(ok) - pr_corr(ok);
gnss.dr = dr(ok) - dr_corr(ok);
gnss.gnss_id = gnss_id(ok);
gnss.sv_id = sv_id(ok);
gnss.sig_id = rov_proc.signal.internal.sig_id(ok);
gnss.accs_id = rov_proc.signal.internal.access_id(ok);
gnss.pr_noise = prNoise(ok);
gnss.dr_noise = drNoise(ok);
gnss.sat_pos_E = [sat_pos_x(ok) sat_pos_y(ok) sat_pos_z(ok)];
gnss.sat_vel_E = [sat_vel_x(ok) sat_vel_y(ok) sat_vel_z(ok)];
gnss.sat_bias = sat_bias(ok);
gnss.sat_drift = sat_drift(ok);
elev_temp = rov_proc.satellite.internal.sat_elev_N(sv_idx,:);
gnss.elev = elev_temp(ok);
azim_temp = rov_proc.satellite.internal.sat_az_N(sv_idx,:);
gnss.azim = azim_temp(ok);
gnss.wavelength = wavelength(ok);
gnss.cno = cno(ok);
z = zeros(size(gnss.sv_id));
gnss.xtId = codeSVIDs(gnss.sv_id, gnss.gnss_id, gnss.sig_id, z);
gnss.xtIdSv = codeSVIDs(gnss.sv_id, gnss.gnss_id, z, z);

% perform any validation
gnss.Validate();

end

function dr_noise = DopplerCnoModel(cno)
% copy of Doppler model from nav_var_freqDevEmp()
% todo: move this somewhere else

dr_noise = nan(size(cno));
dr_noise(cno > 50) = 0.05;
ix = (cno <= 50) & (cno > 40);
dr_noise(ix) = 0.05 + (50 - cno(ix))*0.015;
ix = (cno <= 40) & (cno > 25);
dr_noise(ix) = 0.2 + (40 - cno(ix))*0.085;
ix = (cno <= 25) & (cno > 15);
dr_noise(ix) = 1.475 + (25 - cno(ix))*0.2;
ix = (cno <= 15);
dr_noise(ix) = 3.475 + (15 - cno(ix))*0.3;

end
