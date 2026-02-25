# smartLoc -> MLPKalmannet Schema Report

## Key differences handled
1. Delimiter difference:
- smartLoc files use semicolon (`;`) separators.
- Project files use comma separators.

2. Field name normalization:
- RXM/NAV long descriptive headers are mapped to short project-style names.

3. Unit normalization:
- Doppler from Hz (`doMes`) is converted to m/s (`dr_mps`) via `wavelength_m`.
- Doppler std from Hz (`doStdev`) is converted to m/s (`dr_noise_mps`) via `wavelength_m`.
- GT lon/lat covariance in degree is approximated to EN meters (`gt_pos_std_N_e_m`, `gt_pos_std_N_n_m`).

4. dr sign convention:
- Enforced project convention: `dr_mps = doMes * wavelength_m`.
- With smartLoc sign definition, this aligns `dr_mps` opposite to geometric range-rate.

5. Satellite state enrichment:
- Source nav file: `D:\MLPkalmannet\Data\smartLoc (TU Chemnitz) urban GNSS dataset\berlin2_gendarmenmarkt\aux_nav\BRDC00WRD_U_20161580000_01D_MN.rnx`
- Added per-observation `sat_pos/vel/bias/drift + elev/azim` via BRDC ephemeris propagation.

6. Static relabel:
- `is_static` is recomputed by `speed_mps < 0.1`.

## Assumptions
1. Signal ID is unavailable in RXM-RAWX:
- `sig_id = 0`.

2. Access ID is unavailable:
- `accs_id = 0`.

3. Wavelength inference by constellation:
- GPS/GAL/QZSS: L1/E1 (1575.42 MHz)
- BeiDou: B1I (1561.098 MHz)
- GLONASS: L1 + `freqId` slot (1602 MHz + k*0.5625 MHz)

4. GT velocity vector:
- Only scalar speed + heading are given.
- E/N velocity is inferred from heading (0 rad = East, CCW positive), then rotated to ECEF.

## Notes
- `trans_NAV-POSLLH.csv` is epoch-level GT; it additionally contains epoch-mean sat-state summary columns.
