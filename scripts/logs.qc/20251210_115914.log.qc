2025-12-10 11:59:14 | INFO | parser:log_final_config | Final configuration state:

------------------------------------------------------------------------input_ms-------------------------------------------------------------------------
input_ms.path                                                                                                                           sim_full_chain.ms
input_ms.data_column                                                                                                                                 DATA
input_ms.sigma_column                                                                                                                                None
input_ms.weight_column                                                                                                                               None
input_ms.time_chunk                                                                                                                                     0
input_ms.freq_chunk                                                                                                                                     0
input_ms.is_bda                                                                                                                                     False
input_ms.group_by                                                                                             ['SCAN_NUMBER', 'FIELD_ID', 'DATA_DESC_ID']
input_ms.select_corr                                                                                                                                 None
input_ms.select_fields                                                                                                                                 []
input_ms.select_ddids                                                                                                                                  []
input_ms.select_uv_range                                                                                                                       [0.0, 0.0]
---------------------------------------------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------input_model-----------------------------------------------------------------------
input_model.recipe                                                                                                                             MODEL_DATA
input_model.beam                                                                                                                                     None
input_model.beam_l_axis                                                                                                                                 X
input_model.beam_m_axis                                                                                                                                 Y
input_model.invert_uvw                                                                                                                               True
input_model.source_chunks                                                                                                                             500
input_model.apply_p_jones                                                                                                                           False
---------------------------------------------------------------------------------------------------------------------------------------------------------

-------------------------------------------------------------------------output--------------------------------------------------------------------------
output.gain_directory                                                                                                                            gains.qc
output.log_directory                                                                                                                              logs.qc
output.log_to_terminal                                                                                                                               True
output.overwrite                                                                                                                                     True
output.products                                                                                                                                      None
output.columns                                                                                                                                       None
output.flags                                                                                                                                         True
output.apply_p_jones_inv                                                                                                                            False
output.subtract_directions                                                                                                                           None
output.net_gains                                                                                                                                     None
output.compute_baseline_corrections                                                                                                                 False
output.apply_baseline_corrections                                                                                                                   False
---------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------mad_flags------------------------------------------------------------------------
mad_flags.enable                                                                                                                                    False
mad_flags.whitening                                                                                                                              disabled
mad_flags.threshold_bl                                                                                                                                5.0
mad_flags.threshold_global                                                                                                                           10.0
mad_flags.max_deviation                                                                                                                               0.0
mad_flags.use_off_diagonals                                                                                                                         False
---------------------------------------------------------------------------------------------------------------------------------------------------------

-------------------------------------------------------------------------solver--------------------------------------------------------------------------
solver.terms                                                                                                                                   ['K', 'G']
solver.iter_recipe                                                                                                                               [20, 10]
solver.propagate_flags                                                                                                                               True
solver.robust                                                                                                                                       False
solver.threads                                                                                                                                          1
solver.convergence_fraction                                                                                                                          0.99
solver.convergence_criteria                                                                                                                         1e-06
solver.reference_antenna                                                                                                                                0
---------------------------------------------------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------dask---------------------------------------------------------------------------
dask.threads                                                                                                                                         None
dask.workers                                                                                                                                            1
dask.address                                                                                                                                         None
dask.scheduler                                                                                                                                    threads
dask.scheduler_plugin                                                                                                                                True
---------------------------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------K----------------------------------------------------------------------------
K.type                                                                                                                                              delay
K.solve_per                                                                                                                                       antenna
K.direction_dependent                                                                                                                               False
K.pinned_directions                                                                                                                                   [0]
K.time_interval                                                                                                                                         0
K.freq_interval                                                                                                                                         0
K.load_from                                                                                                                                          None
K.interp_mode                                                                                                                                        reim
K.interp_method                                                                                                                                  2dlinear
K.respect_scan_boundaries                                                                                                                            True
K.initial_estimate                                                                                                                                   True
---------------------------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------G----------------------------------------------------------------------------
G.type                                                                                                                                            complex
G.solve_per                                                                                                                                       antenna
G.direction_dependent                                                                                                                               False
G.pinned_directions                                                                                                                                   [0]
G.time_interval                                                                                                                                         0
G.freq_interval                                                                                                                                         0
G.load_from                                                                                                                                          None
G.interp_mode                                                                                                                                        reim
G.interp_method                                                                                                                                  2dlinear
G.respect_scan_boundaries                                                                                                                            True
G.initial_estimate                                                                                                                                  False
---------------------------------------------------------------------------------------------------------------------------------------------------------


2025-12-10 11:59:14 | INFO | preprocess:transcribe_recipe | The following model sources were obtained from --input-model-recipe: 
   Columns: {'MODEL_DATA'} 
   Sky Models: None
2025-12-10 11:59:14 | INFO | ms_handler:read_xds_list | Antenna table indicates 27 antennas were present for this observation.
2025-12-10 11:59:14 | INFO | ms_handler:read_xds_list | Polarization table indicates 4 correlations are present in the measurement set - ['RR', 'RL', 'LR', 'LL'].
2025-12-10 11:59:14 | INFO | ms_handler:read_xds_list | Field table indicates phase centre is at (-2.7439275283927276 0.5324852164475642).
2025-12-10 11:59:15 | INFO | ms_handler:write_xds_list | Outputs will be written to FLAG, FLAG_ROW.
2025-12-10 11:59:15 | SUCCESS | executor:_execute | 0.80 seconds taken to build graph.
2025-12-10 11:59:15 | INFO | executor:_execute | Computation starting. Please be patient - log messages will only appear per completed chunk.
2025-12-10 11:59:15 | INFO | logging:log_chisq | 
    FLD: 0 SPW: 0 SCN: 1 T_CHUNK: 0 F_CHUNK: 0 CHISQ: 214.58 -> 35.30
2025-12-10 11:59:15 | SUCCESS | executor:_execute | 0.86 seconds taken to execute graph.
2025-12-10 11:59:15 | INFO | logging:log_summary_stats | 
Final post-solve chi-squared summary, colourised by deviation from the mean:
 <= 3*sigma   <= 5*sigma   <= 10*sigma   > 10*sigma       
  CHUNK  FLD0   
         SPW0   
         SCN1   
    
  T0F0   35.30  

