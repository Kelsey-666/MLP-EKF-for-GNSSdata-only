% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.
% 
function [ Ren ] = ecef2nedRot( varargin )
% Transformation matrix from ECEF to NED, latitude and
% longitude are in degrees unless a 3rd argument is defined

latitude = varargin{1};
longitude = varargin{2};
if(length(varargin) < 3)
    latitude = latitude * pi/180;
    longitude = longitude * pi/180;
end
coslat=cos(latitude);
sinlat=sin(latitude);
coslon=cos(longitude);
sinlon=sin(longitude);

Ren=[-sinlat*coslon, -sinlat*sinlon, coslat;...
     -sinlon, coslon, 0.;...
     -coslat*coslon , -coslat*sinlon, -sinlat];

end
