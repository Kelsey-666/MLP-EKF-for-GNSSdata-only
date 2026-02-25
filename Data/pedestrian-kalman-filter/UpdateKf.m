% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

function [kf_stateOut, dx, Sinnov, zres, Sres] = UpdateKf(kf_state, ...
    H, z, R, use_fde)
% Implement the Kalman filter update equations either as a full block, or
% sequentially
x = kf_state.x;
P = kf_state.P;

[x, P, dx, Sinnov, zres, Sres] = UpdateKfEqns(x, P, H, z, R, use_fde);

kf_stateOut = kf_state;
kf_stateOut.x = x;
kf_stateOut.P = P;

end

%% Implement Kalman filter update equations
function [x, P, dx, Sinnov, zres, Sres] = UpdateKfEqns(x, P, H_in, z_in, ...
    R_in, use_fde)
% update the Kalman filter, we assume the inputs are sparse

% intialise before removing nans
keep_ix = find(~isnan(z_in));

% store the states and covariance in case we need to recalculate the update
x_saved = x;
P_saved = P;

iter = 1;
calc_update = true;
while(calc_update && iter <= 4)
    % restore the states if we are recalculating
    if (iter > 1)
        x = x_saved;
        P = P_saved;
    end

    % initialise residual vector with nan's as some measurements may be
    % excluded
    zres = nan(size(z_in));

    % select the measurements
    R = R_in(keep_ix,keep_ix);
    H = H_in(keep_ix,:);
    z = z_in(keep_ix);

    % innovation covariance matrix
    PHt = P * H';
    Sinnov = H * PHt + R;

    % Kalman gain
    K = PHt / Sinnov;
    
    % conventional Kalman filter with optimal gain
    P = P - K * PHt';

    % Improve numerical issues
    P = 0.5 * (P + P');
    % Correction to the states
    dx = K * z;

    % State update
    x = x + dx;

    % Posterior residual
    if (nargout >= 5)
        zres(keep_ix) = z - H * dx;
    end

    % using optimal gain
    Sres = sqrt(diag(R - (speye(length(z)) - H * K) * H * P * H'));

    % Check the residuals and adapt keepix if a measurement fails
    resok = (abs(zres(keep_ix)) < Sres*3.0); % & abs(zres(keep_ix)) < 5;

    if (use_fde && any(~resok))
        % find the worst residual and exlude it
        [~,max_ix] = max(abs(zres(keep_ix)));
        keep_ix(max_ix) = [];
        disp("Excluding measurement "+max_ix);
        iter = iter + 1;
    else
        calc_update = false;
    end
end

end