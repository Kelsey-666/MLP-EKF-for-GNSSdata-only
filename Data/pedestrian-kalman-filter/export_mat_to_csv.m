function [raw_path, gt_path] = export_mat_to_csv(mat_path, out_dir)
%EXPORT_MAT_TO_CSV Export one *_gnss.mat file to:
%  1) raw_observation.csv (per-satellite/per-epoch observation rows)
%  2) gt_processed.csv    (per-epoch ground-truth rows)
%
% Usage:
%   [raw_path, gt_path] = export_mat_to_csv( ...
%       "D:\MLPkalmannet\Data\pedestrian-kalman-filter\Data\xxx_gnss.mat", ...
%       "D:\MLPkalmannet\Data");

if nargin < 1 || strlength(string(mat_path)) == 0
    error("mat_path is required.");
end
if nargin < 2 || strlength(string(out_dir)) == 0
    out_dir = fileparts(mat_path);
end

mat_path = string(mat_path);
out_dir = string(out_dir);

if ~isfile(mat_path)
    error("MAT file not found: %s", mat_path);
end
if ~isfolder(out_dir)
    mkdir(out_dir);
end

% Ensure class folders are visible before loading objects.
script_dir = fileparts(mfilename("fullpath"));
addpath(genpath(script_dir));

% NOTE:
% Do not call "clear classes" here. Inside a function it can clear the
% function workspace itself (including inputs), causing runtime errors.
% If caller previously loaded data with missing class paths, let caller run:
%   clearvars gnss rov_truth
%   clear classes
% then call this function again.

S = load(mat_path, "gnss", "rov_truth");
if ~isfield(S, "gnss")
    error("Variable 'gnss' not found in MAT file: %s", mat_path);
end
gnss = S.gnss;
if isfield(S, "rov_truth")
    rov_truth = S.rov_truth;
else
    rov_truth = [];
end

if isa(rov_truth, "uint32")
    error(["'rov_truth' was loaded as uint32 (class not restored). ", ...
        "Ensure Structs/Truth.m is on path, then rerun this function."]);
end

num_obs = numel(colvec(gnss.epoch));
num_epoch = numel(colvec(gnss.tow));
epoch_idx = colvec(gnss.epoch);

week_epoch = colvec(gnss.week);
tow_epoch = colvec(gnss.tow);
week_obs = pick_by_epoch(week_epoch, epoch_idx, NaN);
tow_obs = pick_by_epoch(tow_epoch, epoch_idx, NaN);
time_gps_s_obs = tow_obs + week_obs .* 604800.0;

% Epoch truth from gnss first, then fallback to rov_truth.
posE_epoch = to_epoch3(getfield_or(gnss, "pos_E", []), num_epoch); %#ok<GFLD>
velE_epoch = to_epoch3(getfield_or(gnss, "vel_E", []), num_epoch); %#ok<GFLD>
posStdN_epoch = to_epoch3(getfield_or(gnss, "pos_std_N", []), num_epoch); %#ok<GFLD>

if all(isnan(posE_epoch), "all") && ~isempty(rov_truth)
    posE_epoch = to_epoch3(getfield_or(rov_truth, "pos_E", []), num_epoch); %#ok<GFLD>
end
if all(isnan(velE_epoch), "all") && ~isempty(rov_truth)
    velE_epoch = to_epoch3(getfield_or(rov_truth, "vel_E", []), num_epoch); %#ok<GFLD>
end
if all(isnan(posStdN_epoch), "all") && ~isempty(rov_truth)
    posStdN_epoch = to_epoch3(getfield_or(rov_truth, "pos_std_N", []), num_epoch); %#ok<GFLD>
end

is_static_epoch = nan(num_epoch, 1);
if ~isempty(rov_truth)
    tmp = to_epoch1(getfield_or(rov_truth, "is_static", []), num_epoch); %#ok<GFLD>
    if ~all(isnan(tmp))
        is_static_epoch = tmp;
    end
end

% Observation arrays
gnss_id = colvec(gnss.gnss_id);
sv_id = colvec(gnss.sv_id);
sig_id = colvec(getfield_or(gnss, "sig_id", nan(size(epoch_idx)))); %#ok<GFLD>
accs_id = colvec(getfield_or(gnss, "accs_id", nan(size(epoch_idx)))); %#ok<GFLD>

pr_m = colvec(gnss.pr);
dr_mps = colvec(gnss.dr);
pr_noise_m = colvec(getfield_or(gnss, "pr_noise", nan(size(epoch_idx)))); %#ok<GFLD>
dr_noise_mps = colvec(getfield_or(gnss, "dr_noise", nan(size(epoch_idx)))); %#ok<GFLD>
iono_m = colvec(getfield_or(gnss, "iono", nan(size(epoch_idx)))); %#ok<GFLD>

sat_pos = to_obs3(getfield_or(gnss, "sat_pos_E", []), num_obs); %#ok<GFLD>
sat_vel = to_obs3(getfield_or(gnss, "sat_vel_E", []), num_obs); %#ok<GFLD>
sat_bias_m = colvec(getfield_or(gnss, "sat_bias", nan(size(epoch_idx)))); %#ok<GFLD>
sat_drift_mps = colvec(getfield_or(gnss, "sat_drift", nan(size(epoch_idx)))); %#ok<GFLD>

elev_rad = colvec(getfield_or(gnss, "elev", nan(size(epoch_idx)))); %#ok<GFLD>
azim_rad = colvec(getfield_or(gnss, "azim", nan(size(epoch_idx)))); %#ok<GFLD>
elev_deg = rad2deg(elev_rad);
azim_deg = rad2deg(azim_rad);
cno = colvec(getfield_or(gnss, "cno", nan(size(epoch_idx)))); %#ok<GFLD>
wavelength_m = colvec(getfield_or(gnss, "wavelength", nan(size(epoch_idx)))); %#ok<GFLD>

% Broadcast epoch truth to observation rows.
gt_pos_obs = pick3_by_epoch(posE_epoch, epoch_idx);
gt_vel_obs = pick3_by_epoch(velE_epoch, epoch_idx);
gt_std_obs = pick3_by_epoch(posStdN_epoch, epoch_idx);
is_static_obs = pick_by_epoch(is_static_epoch, epoch_idx, NaN);

raw_tbl = table();
raw_tbl.obs_index = (1:num_obs).';
raw_tbl.epoch = epoch_idx;
raw_tbl.week = week_obs;
raw_tbl.tow = tow_obs;
raw_tbl.time_gps_s = time_gps_s_obs;
raw_tbl.gnss_id = gnss_id;
raw_tbl.sv_id = sv_id;
raw_tbl.sig_id = sig_id;
raw_tbl.accs_id = accs_id;
raw_tbl.pr_m = pr_m;
raw_tbl.dr_mps = dr_mps;
raw_tbl.pr_noise_m = pr_noise_m;
raw_tbl.dr_noise_mps = dr_noise_mps;
raw_tbl.sat_pos_E_x_m = sat_pos(:, 1);
raw_tbl.sat_pos_E_y_m = sat_pos(:, 2);
raw_tbl.sat_pos_E_z_m = sat_pos(:, 3);
raw_tbl.sat_vel_E_x_mps = sat_vel(:, 1);
raw_tbl.sat_vel_E_y_mps = sat_vel(:, 2);
raw_tbl.sat_vel_E_z_mps = sat_vel(:, 3);
raw_tbl.sat_bias_m = sat_bias_m;
raw_tbl.sat_drift_mps = sat_drift_mps;
raw_tbl.elev_rad = elev_rad;
raw_tbl.azim_rad = azim_rad;
raw_tbl.elev_deg = elev_deg;
raw_tbl.azim_deg = azim_deg;
raw_tbl.cno = cno;
raw_tbl.iono_m = iono_m;
raw_tbl.wavelength_m = wavelength_m;
raw_tbl.gt_pos_E_x_m = gt_pos_obs(:, 1);
raw_tbl.gt_pos_E_y_m = gt_pos_obs(:, 2);
raw_tbl.gt_pos_E_z_m = gt_pos_obs(:, 3);
raw_tbl.gt_vel_E_x_mps = gt_vel_obs(:, 1);
raw_tbl.gt_vel_E_y_mps = gt_vel_obs(:, 2);
raw_tbl.gt_vel_E_z_mps = gt_vel_obs(:, 3);
raw_tbl.gt_pos_std_N_n_m = gt_std_obs(:, 1);
raw_tbl.gt_pos_std_N_e_m = gt_std_obs(:, 2);
raw_tbl.gt_pos_std_N_d_m = gt_std_obs(:, 3);
raw_tbl.is_static = is_static_obs;

% Per-epoch GT
llh = nan(num_epoch, 3);
valid_pos = all(isfinite(posE_epoch), 2);
if any(valid_pos)
    llh_tmp = ecef2llh(posE_epoch(valid_pos, :).');
    llh(valid_pos, :) = llh_tmp.';
end
speed_mps = sqrt(sum(velE_epoch.^2, 2));
if all(isnan(is_static_epoch))
    is_static_epoch = double(speed_mps < 0.2);
end

gt_tbl = table();
gt_tbl.epoch = (1:num_epoch).';
gt_tbl.week = week_epoch;
gt_tbl.tow = tow_epoch;
gt_tbl.time_gps_s = tow_epoch + week_epoch .* 604800.0;
gt_tbl.gt_pos_E_x_m = posE_epoch(:, 1);
gt_tbl.gt_pos_E_y_m = posE_epoch(:, 2);
gt_tbl.gt_pos_E_z_m = posE_epoch(:, 3);
gt_tbl.gt_vel_E_x_mps = velE_epoch(:, 1);
gt_tbl.gt_vel_E_y_mps = velE_epoch(:, 2);
gt_tbl.gt_vel_E_z_mps = velE_epoch(:, 3);
gt_tbl.gt_lat_deg = llh(:, 1);
gt_tbl.gt_lon_deg = llh(:, 2);
gt_tbl.gt_h_m = llh(:, 3);
gt_tbl.gt_pos_std_N_n_m = posStdN_epoch(:, 1);
gt_tbl.gt_pos_std_N_e_m = posStdN_epoch(:, 2);
gt_tbl.gt_pos_std_N_d_m = posStdN_epoch(:, 3);
gt_tbl.speed_mps = speed_mps;
gt_tbl.is_static = is_static_epoch;

raw_path = fullfile(out_dir, "raw_observation.csv");
gt_path = fullfile(out_dir, "gt_processed.csv");
writetable(raw_tbl, raw_path);
writetable(gt_tbl, gt_path);

fprintf("Export done.\n");
fprintf("  raw_observation.csv: %s (rows=%d)\n", raw_path, height(raw_tbl));
fprintf("  gt_processed.csv   : %s (rows=%d)\n", gt_path, height(gt_tbl));

end

function x = colvec(v)
if isempty(v)
    x = [];
else
    x = double(v(:));
end
end

function v = getfield_or(s, field_name, default_v)
if isempty(s) || ~isstruct(s) && ~isobject(s)
    v = default_v;
    return;
end
has_field = false;
if isobject(s)
    has_field = isprop(s, field_name);
elseif isstruct(s)
    has_field = isfield(s, field_name);
end
if has_field
    v = s.(field_name);
else
    v = default_v;
end
end

function out = to_epoch3(a, n)
out = nan(n, 3);
if isempty(a)
    return;
end
a = double(a);
if size(a, 2) == 3
    k = min(n, size(a, 1));
    out(1:k, :) = a(1:k, :);
elseif size(a, 1) == 3
    k = min(n, size(a, 2));
    out(1:k, :) = a(:, 1:k).';
end
end

function out = to_obs3(a, n)
out = nan(n, 3);
if isempty(a)
    return;
end
a = double(a);
if size(a, 2) == 3
    k = min(n, size(a, 1));
    out(1:k, :) = a(1:k, :);
elseif size(a, 1) == 3
    k = min(n, size(a, 2));
    out(1:k, :) = a(:, 1:k).';
end
end

function out = to_epoch1(a, n)
out = nan(n, 1);
if isempty(a)
    return;
end
a = double(a(:));
k = min(n, numel(a));
out(1:k) = a(1:k);
end

function out = pick_by_epoch(epoch_arr, epoch_idx, default_v)
out = default_v * ones(numel(epoch_idx), 1);
if isempty(epoch_arr)
    return;
end
ok = isfinite(epoch_idx) & epoch_idx >= 1 & epoch_idx <= numel(epoch_arr);
ii = round(epoch_idx(ok));
out(ok) = epoch_arr(ii);
end

function out = pick3_by_epoch(epoch_arr3, epoch_idx)
out = nan(numel(epoch_idx), 3);
if isempty(epoch_arr3)
    return;
end
ok = isfinite(epoch_idx) & epoch_idx >= 1 & epoch_idx <= size(epoch_arr3, 1);
ii = round(epoch_idx(ok));
out(ok, :) = epoch_arr3(ii, :);
end
