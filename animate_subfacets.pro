PRO ANIMATE_SUBFACETS, config_path, facet_index
  ; Default parameters if not provided
  IF N_ELEMENTS(config_path) EQ 0 THEN config_path = 'data/config/example_config.yaml'
  IF N_ELEMENTS(facet_index) EQ 0 THEN facet_index = 37
  
  ; --- Setup ---
  cfg = READ_CONFIG(config_path)
  sim = CREATE_SIMULATION(cfg)
  
  ; Read the global shape model
  shape = READ_SHAPE_MODEL(cfg.path_to_shape_model_file, $
                         sim.timesteps_per_day, $
                         sim.n_layers, $
                         sim.max_days, $
                         cfg.calculate_energy_terms)
  
  ; Generate depressions and solve subfacet conduction
  solver = CREATE_TEMP_SOLVER(cfg.temp_solver)
  
  FOR i=0, N_ELEMENTS(shape)-1 DO BEGIN
    f = shape[i]
    f.GENERATE_SPHERICAL_DEPRESSION, cfg, sim
    solver->INITIALIZE_TEMPERATURES, f.depression_thermal_data, sim, cfg
    f.depression_temperature_result = solver->SOLVE(f.depression_thermal_data, $
                                                  f.sub_facets, $
                                                  sim, $
                                                  cfg)
  ENDFOR

  ; Pick our facet
  parent = shape[facet_index]
  subs = parent.depression_temperature_result.final_day_temperatures ; (N_sub x T)
  dims = SIZE(subs, /DIMENSIONS)
  N = dims[0]
  T = dims[1]

  ; Rebuild world-space triangles for subfacets
  mesh_entries = Facet._canonical_subfacet_mesh
  parent_radius = SQRT(parent.area/!PI)
  scale = cfg.kernel_dome_radius_factor * parent_radius
  
  verts = FLTARR(3, 3*N)
  faces = LONARR(4, N)
  vidx = 0
  
  FOR j=0, N_ELEMENTS(mesh_entries)-1 DO BEGIN
    ; local → world transform
    tri = mesh_entries[j].vertices * scale
    world_tri = TRANSPOSE(parent.dome_rotation ## TRANSPOSE(tri)) + REBIN(parent.position, 3, 3)
    
    verts[*, vidx:vidx+2] = world_tri
    faces[*, j] = [3, vidx, vidx+1, vidx+2]
    vidx += 3
  ENDFOR

  ; Save to temporary STL file for visualization
  tmp_stl = 'subfacet_temp.stl'
  WRITE_STL, tmp_stl, verts, faces
  
  ; Call the same animate_model routine with our mesh
  ANIMATE_MODEL, $
    tmp_stl, $                       ; path_to_shape_model_file
    subs, $                          ; plotted_variable_array
    sim.rotation_axis, $
    sim.sunlight_direction, $
    sim.timesteps_per_day, $
    sim.solar_distance_au, $
    sim.rotation_period_hours, $
    COLOR_TABLE=74, $                ; IDL's "rainbow" colormap (similar to coolwarm)
    TITLE=STRING(FORMAT='("Sub-facet Temps for facet %d")', facet_index), $
    AXIS_LABEL='Temperature (K)', $
    N_FRAMES=T, $
    SAVE_ANIMATION=0, $
    ANIMATION_FILENAME=STRING(FORMAT='("facet_%d_subfacets.gif")', facet_index), $
    BACKGROUND_COLOR=[255, 255, 255]  ; white
    
  ; Clean up temporary file
  FILE_DELETE, tmp_stl, /QUIET
END

;--------------------------------------
; Helper functions - these would normally be in separate files
;--------------------------------------

FUNCTION READ_CONFIG, config_path
  ; Implementation would read YAML config file
  ; Using IDL's YAML parsing capabilities
  yaml_data = YAML_PARSE(config_path)
  RETURN, yaml_data
END

FUNCTION CREATE_SIMULATION, cfg
  ; Create simulation object from config
  sim = {timesteps_per_day: 0L, $
         n_layers: 0L, $
         max_days: 0L, $
         rotation_axis: FLTARR(3), $
         sunlight_direction: FLTARR(3), $
         solar_distance_au: 0.0, $
         rotation_period_hours: 0.0}
         
  ; Initialize values from config
  sim.timesteps_per_day = 24L ; Example - would be calculated from cfg
  sim.n_layers = cfg.n_layers
  sim.max_days = cfg.max_days
  sim.rotation_axis = [0, 0, 1] ; Example - would be calculated from RA/DEC
  sim.sunlight_direction = cfg.sunlight_direction
  sim.solar_distance_au = cfg.solar_distance_au
  sim.rotation_period_hours = cfg.rotation_period_hours
  
  RETURN, sim
END

FUNCTION READ_SHAPE_MODEL, path, timesteps_per_day, n_layers, max_days, calc_energy
  ; Read STL file and create facet objects
  facets = READ_STL(path)
  
  ; Create facet objects with properties needed for thermal modeling
  ; In real implementation, would process the STL data
  
  RETURN, facets
END

FUNCTION CREATE_TEMP_SOLVER, solver_type
  ; Factory function to create temperature solver
  CASE STRUPCASE(solver_type) OF
    'TEMPEST_STANDARD': RETURN, OBJ_NEW('TempestStandardSolver')
    'THERMPROJRS_LIKE': RETURN, OBJ_NEW('ThermprojrsSolver')
    ELSE: RETURN, OBJ_NEW('TempestStandardSolver')
  ENDCASE
END

PRO WRITE_STL, filename, vertices, faces
  ; Implementation of STL file writing
  ; In real implementation would convert vertices and faces to STL format
  OPENW, lun, filename, /GET_LUN
  ; Write STL header and data
  CLOSE, lun
  FREE_LUN, lun
END

PRO ANIMATE_MODEL, model_file, variable_array, rotation_axis, sunlight_direction, $
                  timesteps_per_day, solar_distance, rotation_period, $
                  COLOR_TABLE=color_table, TITLE=title, AXIS_LABEL=axis_label, $
                  N_FRAMES=n_frames, SAVE_ANIMATION=save_animation, $
                  ANIMATION_FILENAME=animation_filename, BACKGROUND_COLOR=background_color
                  
  ; Implementation of IDL animation
  ; This would use IDL's graphics capabilities to:
  ; 1. Load the 3D model
  ; 2. Set up color mapping for temperature data
  ; 3. Create a window or device for output
  ; 4. Loop through timesteps, updating the model coloring
  ; 5. Optionally save frames to GIF
  
  ; Example simplified implementation:
  model = OBJ_NEW('IDLgrModel')
  ; Load STL into model
  
  ; Create window or save to file based on parameters
  IF KEYWORD_SET(save_animation) THEN BEGIN
    ; Set up animation saving
  ENDIF ELSE BEGIN
    window = WINDOW(DIMENSIONS=[800, 600], TITLE=title)
  ENDELSE
  
  ; Loop through frames
  FOR i=0, n_frames-1 DO BEGIN
    ; Update model with coloring based on variable_array[*,i]
    ; Update view angle based on timestep
    ; Render and save or display frame
  ENDFOR
  
  ; Clean up
  OBJ_DESTROY, model
END

; Run the program if executed directly
IF N_ELEMENTS(ROUTINE_NAMES(/MAIN)) GT 0 THEN ANIMATE_SUBFACETS 