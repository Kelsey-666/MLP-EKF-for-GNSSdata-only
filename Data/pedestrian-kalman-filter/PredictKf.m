% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

function [kf_stateOut, Phi] = PredictKf(kf_state, dt)
%% Predict the Kalman filter

state_indx = kf_state.state_indx;

% initial values
q = zeros(kf_state.num_states,1); % diagonal of Q
PhiDiag = ones(kf_state.num_states,1); % diagonal of Phi

% position and velocity process noise using constant acceleration model
accel_h = 5;     % horizontal acceleration intensity
accel_v = 0.05;  % vertical acceleration intensity
llh = ecef2llh(kf_state.x(state_indx.pos));
n2e = ecef2nedRot(llh(1), llh(2))';
qs = n2e * diag([accel_h accel_h accel_v]) * n2e';
dt2 = dt * dt;
dt3 = dt2 * dt;
dt4 = dt3 * dt;

% Qpv is added at the end of this function as its dense
Qpv = [0.25*dt4*qs 0.5*dt3*qs; 0.5*dt3*qs' dt2*qs];

% clock bias using model parameters from smor
h_0 = 17.e-3;
h_2 = 10*h_0;
Sf = 2*h_0;
Sg = 8*pi^2*h_2;
dt2by2 = dt*dt/2;      % dt^2/2
dt3by6 = 2*dt*dt2by2/3;	% dt^3/6

if (~isempty(state_indx.ibb))
    % inter-band biases
    q(state_indx.ibb) = 0.01^2 * dt;
end
if (~isempty(state_indx.inter_gnss))
    % inter-GNSS biases
    q(state_indx.inter_gnss) = 0.01^2 * dt;
end

% Form Phi and Q and add constant velocity terms
Phi = sparse([1:kf_state.num_states 1:3],[1:kf_state.num_states 4:6],[PhiDiag; dt; dt; dt]);
Q = sparse(1:kf_state.num_states,1:kf_state.num_states,q);
Q(state_indx.posvel,state_indx.posvel) = Qpv;

% Add clock model to Phi and Q
Phi(state_indx.clock(1),state_indx.clock(2)) = dt;
Q(state_indx.clock,state_indx.clock) = [Sf*dt+Sg*dt3by6 Sf*dt2by2;...
     Sf*dt2by2  Sf*dt];

kf_stateOut = kf_state;
kf_stateOut.P = Phi * kf_state.P * Phi' + Q;
kf_stateOut.x = Phi * kf_state.x;

end
