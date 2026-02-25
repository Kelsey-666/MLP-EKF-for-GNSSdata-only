% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

function [pred_pr, pred_dr] = ...
    CalcPredictedObs(satellite, signal, phase_smooth)
%% calculate predicted PR and phase.
% Args:
%  satellite:    satellite class for receiver
%  signal:       signal class for receiver
%  solver_proc:  SolverProc class for receiver
%  heading:      heading of antenna
%  phase_smooth: true if phase smoothing should be applied
%
% Returns:
%
%  pred_pr    : predicted pseudorange [m]
%  pred_phase : predicted carrier phase [cycles]

% form predicted observation
trop = satellite.model.tropo_delay.map_dry .* ...
    satellite.model.tropo_delay.zenith_dry + ...
    satellite.model.tropo_delay.map_wet .* ...
    satellite.model.tropo_delay.zenith_wet;

% calculate the iono model per signal
if (~isempty(satellite.model.gim))
    iono_scale = 40.3e16./(signal.internal.freq.^2);
    iono_sig_m = iono_scale .* satellite.model.gim(signal.internal.sv_idx,:);
else
    iono_sig_m = 0;
end

%% PR
pred_pr = satellite.internal.linearised.pr(signal.internal.sv_idx, :) + ...
    trop(signal.internal.sv_idx,:) + iono_sig_m;

if (~isempty(signal.model.sat_pco))
    pred_pr = pred_pr - signal.model.sat_pco;
end

if (~isempty(signal.model.pco))
    pred_pr = pred_pr + signal.model.pco;
end

if (~isempty(signal.model.sat_pr_bias))
    % PPP corrections
    pred_pr = pred_pr + signal.model.sat_pr_bias;
    
    if (phase_smooth && ~isempty(signal.model.sat_phase_bias))
        % if using phase smoothing we use both the satellite PR bias to
        % model the satellite dependent delay, but also the phase
        % bias to model e.g. GPS L5 line bias
        pred_pr = pred_pr + signal.model.sat_phase_bias;
    end
end

% apply code bias variation if we have values
if (~isempty(signal.model.cbv))
    pred_pr = pred_pr + signal.model.cbv;
end

% apply either phase smoothing or BeiDou biases
if (phase_smooth)
    pred_pr = pred_pr + solver_proc.signal.pr_multipath;
else
    if (~isempty(signal.model.sat_nadir_bias))
        pred_pr = pred_pr - signal.model.sat_nadir_bias;
    end
end

%% Doppler
if (nargout > 1)
    % predict Doppler using geometry that already has the satellite clock
    % rate applied
    pred_dr = satellite.internal.linearised.doppler(signal.internal.sv_idx, :);
end

end
