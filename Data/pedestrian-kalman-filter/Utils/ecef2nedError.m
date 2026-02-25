% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

function [dpos_ned, dpos_ecef, dvel_ned, dvel_ecef] = ...
    ecef2nedError(pos_E, truth_pos_E, vel_E, truth_vel_E)
%% Converts the position difference from ECEF to NED
%% function [dpos_ned, dpos_ecef] = ecef2nedError(pos_E, truth_E)
% inputs:
%   pos_E = position in ECEF (n x 3) array
%   truth_pos_E = reference position in ECEF (n x 3) array
%   vel_E = velocity in ECEF (n x 3) array (optional)
%   truth_vel_E = reference velocity in ECEF (n x 3) array (mandatory if
%                 vel_E given)
% output:
%   dpos_ned = difference in position in NED
%   dpos_ecef = difference in position in ECEF
dpos_ecef = pos_E - truth_pos_E;
dpos_ned = nan(size(dpos_ecef));
if (nargin > 2)
    dvel_ecef = vel_E - truth_vel_E;
    dvel_ned = nan(size(dvel_ecef));
end

%% Hack!
%n2n2 = euler2dcm([0;0;(-90+60)*pi/180]);
%warning("Rotating navigation frame for error analysis!");
n2n2 = eye(3);

for i = 1:size(pos_E,1)
    llh = ecef2llh(pos_E(i,:)');
    Ren = ecef2nedRot(llh(1), llh(2));

    dpos_ned(i,:) = n2n2 * Ren * dpos_ecef(i,:)';
    if (nargin > 2)
        dvel_ned(i,:) = n2n2 * Ren * dvel_ecef(i,:)';
    end
end

end
