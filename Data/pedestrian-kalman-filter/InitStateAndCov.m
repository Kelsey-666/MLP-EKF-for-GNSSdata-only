% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

function kf_stateOut = InitStateAndCov(initPos, kf_state)

p = zeros(kf_state.num_states,1); % diagonal of covariance matrix (initial variances)
x = zeros(kf_state.num_states,1);

% initialise states, add an initial 2m bias (stdev) to the position
x(kf_state.state_indx.posvel) = ...
    [(initPos + randn(1,3) * 2) 0 0 0];

% initialise the covariance matrix

% initial position and velocity
p(kf_state.state_indx.posvel) = 1000.^2;

% initial clock
p(kf_state.state_indx.clock(1)) = 1000.^2; % clock bias
p(kf_state.state_indx.clock(2)) = 100.^2;  % clock drift

if (~isempty(kf_state.state_indx.ibb))
    % initial IBB biases
    p(kf_state.state_indx.ibb) = 1000^2;
end

% initial inter-GNSS biases
if (~isempty(kf_state.state_indx.inter_gnss))
    p(kf_state.state_indx.inter_gnss) = 1000^2;
end

kf_stateOut = kf_state;
kf_stateOut.P = sparse(diag(p));
kf_stateOut.x = x;

end