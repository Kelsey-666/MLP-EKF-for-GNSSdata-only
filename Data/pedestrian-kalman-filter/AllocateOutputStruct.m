% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

function out = AllocateOutputStruct(numEpoch,kf_state)
%% Allocate arrays with nan's for output

out.state_indx       = kf_state.state_indx; % Index of the state_indx values
out.x               = nan(numEpoch, kf_state.num_states);
out.std             = nan(numEpoch, kf_state.num_states);
out.stdPosNed       = nan(numEpoch, 3);
out.week            = nan(numEpoch,1);
out.tow             = nan(numEpoch,1);

out.num_obs          = nan(numEpoch,1);
out.numSat          = nan(numEpoch,1);
out.pdop            = nan(numEpoch,1);
out.prRes           = nan(numEpoch, kf_state.num_obs);
out.drRes           = nan(numEpoch, kf_state.num_obs);
out.prInnov         = nan(numEpoch, kf_state.num_obs);

end