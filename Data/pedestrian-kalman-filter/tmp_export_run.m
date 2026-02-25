try
    diary(''D:/MLPkalmannet/Data/export_log.txt'');
    diary on;
    disp(''START_EXPORT'');
    cd(''D:/MLPkalmannet/Data/pedestrian-kalman-filter'');
    [raw_path, gt_path] = export_mat_to_csv(...
        ''D:/MLPkalmannet/Data/pedestrian-kalman-filter/Data/SPG520B00_NavRoadtestSPU_S220421h10_Wear_CN_Urban_NoAid_R3__001_test__rxlog_01_EIP_A_16721_GEBcRQ_FP.ubz_MATLAB_gnss.mat'', ...
        ''D:/MLPkalmannet/Data'');
    disp(raw_path);
    disp(gt_path);
    disp(''END_EXPORT_OK'');
    diary off;
catch ME
    disp(getReport(ME,''extended'',''hyperlinks'',''off''));
    diary off;
    exit(2);
end
exit(0);
