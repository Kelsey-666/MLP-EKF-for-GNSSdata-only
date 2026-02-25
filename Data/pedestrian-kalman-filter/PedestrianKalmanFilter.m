% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

function out = PedestrianKalmanFilter(gnss)

% check length of input data
if (isempty(gnss.gnss_id))
    warning('No input data found');
    return;
end

% identify epochs
temp = [1; 1+find(diff(gnss.epoch)); length(gnss.epoch)+1];
epoch_ix = nan(size(gnss.tow,1),2);
lt = length(temp);
epoch_ix(gnss.epoch(temp(1:lt-1)),1) = temp(1:lt-1);
epoch_ix(gnss.epoch(temp(1:lt-1)),2) = temp(2:lt)-1;
clear temp;
numEpoch = length(gnss.tow);

% calculate unique number of satellites, observations and signals
[kf_state, obs_id, gnss, map] = GetObsMap(gnss);

% storage for output
out = AllocateOutputStruct(numEpoch, kf_state);
out.state_indx = kf_state.state_indx;

% initialise the states
kf_state = InitStateAndCov(gnss.pos_E(1,:), kf_state);
first_epoch = true;

% iterate through the epochs
for epoch = 1:numEpoch
    % store measurement indices
    ix = epoch_ix(epoch,1):epoch_ix(epoch,2);

    % predict the states
    if (~first_epoch)
        dt = gnss.tow(epoch) - gnss.tow(epoch-1);
        kf_state = PredictKf(kf_state, dt);
    end

    % check if we have enough measurements
    if (~isnan(ix(1)))
        uniqueGnss = unique(gnss.gnss_id(ix));
        uniqueSat  = unique(gnss.gnss_id(ix)*256+gnss.sv_id(ix));
        if (length(uniqueSat) < 3 + length(uniqueGnss))
            ix(1) = nan; % invalidate the epoch
        else
            first_epoch = false;
        end
    end

    if (~isnan(ix(1)))
        % form the Pseudorange, carrier phase and Doppler observations and
        % update in block
        [pr, dr, kf_state, out.pdop(epoch)] = ...
            FormObservations(gnss, ix, kf_state, map);

        % store pseudorange innovations
        out.prInnov(epoch,map.chn(ix)) = pr.z;

        % update using Doppler measurements
        [kf_state, dx, ~, dr_res] = UpdateKf(kf_state, dr.H, dr.z, ...
            dr.R, true);

        % adjust the other measurements for the change of state
        pr.z = pr.z - pr.H * dx;

        % store posterior Doppler residuals
        out.drRes(epoch,map.chn(ix)) = dr_res;

        % update the Kalman filter with the pseudoranges
        [kf_state,~,~,pr_res] = UpdateKf(kf_state, pr.H, pr.z, pr.R,true);

        % store pseudorange residuals in the given channel
        out.prRes(epoch,map.chn(ix)) = pr_res;
    end

    % store states
    llh = ecef2llh(kf_state.x(1:3));
    Ren = ecef2nedRot(llh(1), llh(2));

    % store the states and residuals
    out.x(epoch,:) = kf_state.x;
    out.std(epoch,:) = sqrt(diag(kf_state.P));
    out.stdPosNed(epoch,:) = sqrt(diag(Ren * kf_state.P(1:3,1:3) * Ren'));

    out.tow(epoch)  = gnss.tow(epoch);
    out.week(epoch) = gnss.week(epoch);
    if (~isnan(ix(1)))
        out.num_obs(epoch) = length(ix);
        out.numSat(epoch) = ...
            length(unique(gnss.gnss_id(ix)*256+gnss.sv_id(ix)));
    else
        out.num_obs(epoch) = 0;
        out.numSat(epoch) = 0;
    end

    % check for matrix issues, chol returns a flag indicating whether the
    % matrix is positive definite
    [~,check] = chol(kf_state.P);
    if (check ~= 0)
        warning('Positive definite matrix check failed');
    end
end

%remove useless fields
out = cleanStruct(out);

% store indexing information
out.state_indx = kf_state.state_indx;
out.obsId = obs_id;
% done

end
