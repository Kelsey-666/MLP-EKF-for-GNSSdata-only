% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

function [kf_state, obs_id, gnss, map] = GetObsMap(gnss)
% get the required information to map observations to states

kf_state = KFState;
kf_state.state_indx = [];

% unique satellites
xtIdSv = codeSVIDs(gnss.sv_id, gnss.gnss_id, 0, 0);
obs_id.xtIdSv = unique(xtIdSv);
kf_state.num_sv = length(obs_id.xtIdSv);

% unique observations
extId = codeSVIDs(gnss.sv_id, gnss.gnss_id, gnss.sig_id, 0);
obs_id.xtId = unique(extId);
kf_state.num_obs = length(unique(obs_id.xtId));

% unique GNSS for IBB estimation. GPS is not estimated therefore always
% need GPS to be present in the data.
obs_id.interGnss = unique(gnss.gnss_id);
% choose one of the GNSS to the be the reference (choose the smallest)
obs_id.interGnss(obs_id.interGnss == obs_id.interGnss(1)) = [];
kf_state.inter_gnss = length(obs_id.interGnss);

% add IBBs for each constellation
ibb = codeSVIDs(0, gnss.gnss_id, gnss.sig_id, 0);
obs_id.ibb = [];
for gnssId = unique(gnss.gnss_id)'
    % find unique signals
    sigIds = unique(gnss.sig_id(gnss.gnss_id == gnssId));
    % add sigIds
    for i = 2:(length(sigIds))
        obs_id.ibb(end+1) = codeSVIDs(0, gnssId, sigIds(i), 0);
    end
end
kf_state.ibb = length(obs_id.ibb);

% append the extra information to the gnss structure
gnss.xtId = extId;
gnss.xtIdSv = xtIdSv;

% store the mapping between observations and states
[~,map.sv] = ismember(xtIdSv,obs_id.xtIdSv);
[~,map.interGnss] = ismember(gnss.gnss_id,obs_id.interGnss);
[~,map.ibb] = ismember(ibb,obs_id.ibb);
[~,map.chn] = ismember(extId, obs_id.xtId); % unique channel ID for each obs

% store the first index for each group of states and calculate the size of
% the state_indx vector

% position, velocity, clock and clock drift are modelled by default
state_indx.pos = 1:3;
state_indx.vel = 4:6;
state_indx.posvel = 1:6;
state_indx.clock = 7:8;
kf_state.num_states = 8;

% add bias states
if (kf_state.ibb > 0)
    state_indx.ibb = (kf_state.num_states + 1): (kf_state.num_states + kf_state.ibb);
    kf_state.num_states = kf_state.num_states + kf_state.ibb;
else
    state_indx.ibb = [];
end
if (kf_state.inter_gnss > 0)
    state_indx.inter_gnss = (kf_state.num_states + 1): (kf_state.num_states + kf_state.inter_gnss);
    kf_state.num_states = kf_state.num_states + kf_state.inter_gnss;
else
    state_indx.inter_gnss = [];
end

kf_state.state_indx = state_indx;

end
