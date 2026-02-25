% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

function llh = ecef2llh(xyz)
% xyz2llh:	converts from ECEF coordinates to navigation coordinates
%           (latitude, longitude and altitude).
% INPUTS
%	xyz: [X Y Z] 3 x n ECEF coordinates (meters)
%
% OUTPUTS
%	llh: [latitude,longitude,altitude above ellipsoid (deg, deg, meters)]
%   Reference:
%     Title: Direct transformation from geocentric coordinates to geodetic coordinates
%     Authors: Vermeille, H.
%     Publication: Journal of Geodesy, Vol. 76, No. 8, p. 451-454
%     Publication Date:	11/2002

llh = zeros(size(xyz));

a = 6378137;         % semi-major axis in meters
f = 1/298.257223563; % the flattening factor
e2 = f*(2-f);        % square of first eccentricity
e4 = e2 * e2;

% Longitude
llh (2,:) = (180/pi) * atan2(xyz(2,:) , xyz(1,:));

%%
z = xyz(3,:)/a;
p = (sum(xyz(1:2,:).^2)) ./ a^2;
q = (1-e2) * z.^2;
r = (p + q - e4) ./ 6;
s = (e4 * p .* q) ./ (4 * r.^3);
t = (1 + s + sqrt(s .* (2+s))).^ (1/3);
u = r .* (1 + t + ( 1./t ) );
v = sqrt( u.^2 + q .* e4);
w = e2 .* ( u + v - q ) ./ ( 2 .* v );
k = sqrt( u + v + w.^2 ) - w;

D = k .* sqrt(p) ./ (k + e2);

% Latitude
llh(1,:) = (180/pi) * atan2(z , D);

% Altitude
llh(3,:) = a * (k + e2 -1) ./ k .* (sqrt(D.^2 + z.^2));

end