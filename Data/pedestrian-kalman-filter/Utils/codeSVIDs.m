% Copyright (c) 2025 u-blox AG
% 
% All rights reserved. This source code is confidential and proprietary.
% Unauthorized copying, distribution, or use of this code, in whole or in part,
% is strictly prohibited without prior written permission.

function extI = codeSVIDs(svId, gnssId, sigId, accsId)
% function extI = codeSVIDs(svId, gnssId, sigId, accsId)
% calculates an extended Id for a combination of
% satellite/constellation/signal
extI = svId + 256*(gnssId+256*(sigId+256*accsId.*(gnssId == 6)));
extI = extI.*(sigId ~= 255);
end