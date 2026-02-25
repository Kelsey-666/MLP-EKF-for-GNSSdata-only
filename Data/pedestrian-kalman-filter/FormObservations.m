% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

function [pr, dr, kf_state, PDOP] = ...
    FormObservations(gnss, ix, kf_state, map)
%% Form observations for the current epoch

x = kf_state.x;
P = kf_state.P;
% Constants
WGS84_W = (7292115.1467e-11);
SOL = 299792458.0;

% calculate the predicted range and line of sight
user_pos_E = x(kf_state.state_indx.pos);
sat_pos_E = gnss.sat_pos_E(ix,:)';
sat_vel_E = gnss.sat_vel_E(ix,:)';

% satellite position is in ECEF at time of transmission so calculate the
% satellite position using ECEF at time of reception
delta_pos = (gnss.sat_pos_E(ix,:) - user_pos_E')';
theta = WGS84_W * (vecnorm(delta_pos) / SOL);

sat_pos_E = [ sat_pos_E(1,:).*cos(theta) + sat_pos_E(2,:).*sin(theta); ...
             -sat_pos_E(1,:).*sin(theta) + sat_pos_E(2,:).*cos(theta); ...
              sat_pos_E(3,:) ];

% calculate corrected difference in position
delta_pos = (sat_pos_E - user_pos_E);

% predicted geometric range
range = sqrt(sum(delta_pos.^2,1));

% line of sight vectors
negative_los = -(delta_pos' ./ range');

n = length(ix);

% position, velocity, clock
Hpos = [negative_los zeros(n,3) ones(n,1) zeros(n,1)];

% calculate PDOP using unique satellites
[~,uix] = unique(gnss.sv_id(ix)+256*gnss.gnss_id(ix));
A = Hpos(uix,1:3);
if (length(uix) >= 4)
    temp = inv(A'*A);
    PDOP = sqrt(sum(diag(temp(1:3,1:3))));
else
    PDOP = nan;
end

% IBB
row = 1:n;
col = map.ibb(ix);
row(col == 0) = [];
col(col == 0) = [];

% inter-GNSS biases
row2 = 1:n;
col2 = map.interGnss(ix);
row2(col2 == 0) = [];
col2(col2 == 0) = [];
col2 = col2 + kf_state.ibb;

Hbias = sparse([row row2]',[col;col2],1,n,kf_state.ibb+kf_state.inter_gnss);

% store the models (i.e. differential corrections) from the simulation
% The predicted range below is calculated from the non-linear (i.e.
% position) and linear parts (i.e. everything else) of the model
satclk = gnss.sat_bias(ix);

%% store the pseudorange

% pseudorange measurement noise
pr.R = sparse(diag(gnss.pr_noise(ix).^2));

% decide how to model biases
bias_block = Hbias;

% pseudorange design matrix
pr.H = sparse([Hpos bias_block]);

% calculated range
pred = range' + pr.H(:,kf_state.state_indx.clock(1):end) * x(kf_state.state_indx.clock(1):end) - satclk;

% innovation
pr.z = gnss.pr(ix) - pred;

%% Doppler measurement

% predict the velocity
user_vel = x(kf_state.state_indx.vel);
delta_vel = sat_vel_E - user_vel;

range_rate = dot(delta_vel,-negative_los')' - gnss.sat_drift(ix);

dr.H = sparse(length(ix),length(x));
% note - ignore weak relationship with position states
dr.H(:,kf_state.state_indx.vel) = negative_los;
dr.H(:,kf_state.state_indx.clock(2)) = 1;     % clock drift

% note Doppler sign is negative (i.e. frequency is higher when moving
% closer but range rate is smaller)
dr.z = (-gnss.dr(ix) - range_rate) - dr.H(:,kf_state.state_indx.clock(1):end) * x(kf_state.state_indx.clock(1):end);

dr.R = sparse(diag(gnss.dr_noise(ix).^2));

%% check for millisecond jumps in the clock
ms_jump = round(median(pr.z,"omitnan")/(SOL*0.001))*SOL*0.001;
pr.z = pr.z - ms_jump;
x(kf_state.state_indx.clock(1)) = x(kf_state.state_indx.clock(1)) + ms_jump;

if (abs(median(pr.z,"omitnan")) > 1000)
    jump = median(pr.z,"omitnan");
    disp("Adjusting receiver clock: "+num2str(jump)+"m");
    pr.z = pr.z - jump;
    x(kf_state.state_indx.clock(1)) = x(kf_state.state_indx.clock(1)) + jump;
end

if (abs(median(dr.z,"omitnan")) > 100)
    jump = median(dr.z,"omitnan");
    disp("Adjusting receiver clock drift: "+num2str(jump)+"m/s");
    dr.z = dr.z - jump;
    x(kf_state.state_indx.clock(2)) = x(kf_state.state_indx.clock(2)) + jump;
end
kf_state.x = x;
kf_state.P = P;

end