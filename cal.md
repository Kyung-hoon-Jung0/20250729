# Calibration Workflow
```mermaid
%%{init: {
  'theme':'base',
  'themeVariables': {
    'primaryColor':'#e6e6fa',
    'primaryTextColor':'#000',
    'primaryBorderColor':'#9370db',
    'lineColor':'#6b8e23',
    'secondaryColor':'#fffacd',
    'tertiaryColor':'#fff'
  },
  'flowchart': {
    'nodeSpacing': 70,
    'rankSpacing': 30,
    'padding': 10,
    'htmlLabels': true
  }
}}%%
flowchart TB

%% Fixed column widths
%% Left/Right: 420px, Center: 320px

subgraph Exp1[" "]
  direction LR
  I1["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- default_cable_setting</div>"]
  E1["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Time of Flight</div>"]
  O1["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- time_of_flight (ns)<br/>- input_gain_db</div>"]
  I1 --> E1 --> O1
end

subgraph Exp2[" "]
  direction LR
  I2["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- readout_frequency</div>"]
  E2["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Resonator Spectroscopy<br/>(Wide Scan)</div>"]
  O2["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- resonator_frequency</div>"]
  I2 --> E2 --> O2
end

subgraph Exp3[" "]
  direction LR
  I3["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- readout_frequency<br/>- readout_amplitude</div>"]
  E3["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Resonator Spectroscopy<br/>(Individual Scan)</div>"]
  O3["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- resonator_frequency<br/>- readout_power</div>"]
  I3 --> E3 --> O3
end

subgraph Exp4[" "]
  direction LR
  I4["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- pump_frequency<br/>- pump_amplitude</div>"]
  E4["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>TWPA calibration</div>"]
  O4["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- optimal_pump_frequency<br/>- optimal_pump_amplitude</div>"]
  I4 --> E4 --> O4
end

subgraph Exp5[" "]
  direction LR
  I5["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- readout_frequency<br/>- readout_amplitude</div>"]
  E5["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Resonator Spectroscopy<br/>vs Power</div>"]
  O5["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- resonator_frequency<br/>- readout_amplitude</div>"]
  I5 --> E5 --> O5
end

subgraph Exp6[" "]
  direction LR
  I6["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- readout_frequency<br/>- z_flux_amplitude</div>"]
  E6["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Resonator Spectroscopy<br/>vs Qubit Flux</div>"]
  O6["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- joint_UPSS_voltages_set</div>"]
  I6 --> E6 --> O6
end

subgraph Exp7[" "]
  direction LR
  I7["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- readout_frequency<br/>- coupler_flux_amplitude</div>"]
  E7["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Resonator Spectroscopy<br/>vs Coupler Flux</div>"]
  O7["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- coupler_bias_phi_0_over_4</div>"]
  I7 --> E7 --> O7
end

subgraph Exp8[" "]
  direction LR
  I8["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- drive_frequency</div>"]
  E8["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Qubit Spectroscopy</div>"]
  O8["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- qubit_frequency</div>"]
  I8 --> E8 --> O8
end

subgraph Exp9[" "]
  direction LR
  I9["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- drive_frequency<br/>- qubit_flux</div>"]
  E9["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Qubit Spectroscopy<br/>vs Qubit Flux</div>"]
  O9["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- frequency_dispersion<br/>- UPSS_fine_tune</div>"]
  I9 --> E9 --> O9
end

subgraph Exp10[" "]
  direction LR
  I10["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- drive_frequency<br/>- coupler_flux</div>"]
  E10["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Qubit Spectroscopy<br/>vs Coupler Flux</div>"]
  O10["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- frequency_dispersion<br/>- UPSS_fine_tune_for_coupler_flux</div>"]
  I10 --> E10 --> O10
end

subgraph Exp11[" "]
  direction LR
  I11["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- drive_amplitude</div>"]
  E11["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Power Rabi</div>"]
  O11["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- pulse_amplitude</div>"]
  I11 --> E11 --> O11
end

subgraph Exp12[" "]
  direction LR
  I12["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- evolution_time</div>"]
  E12["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Ramsey</div>"]
  O12["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- fine_tuned_qubit_frequency</div>"]
  I12 --> E12 --> O12
end

subgraph Exp13[" "]
  direction LR
  I13["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- num_of_pi_repettion<br/>- drive_amplitude</div>"]
  E13["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Phase Error Amp<br/>(DRAG)</div>"]
  O13["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- DRAG_coefficient</div>"]
  I13 --> E13 --> O13
end

subgraph Exp14[" "]
  direction LR
  I14["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- num_of_pi_repettion<br/>- drive_amplitude</div>"]
  E14["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Error Amp Rabi</div>"]
  O14["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- single_qubit_gate_params</div>"]
  I14 --> E14 --> O14
end

subgraph Exp15[" "]
  direction LR
  I15["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- readout_frequency<br/>- readout_amplitude</div>"]
  E15["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Readout Optimization</div>"]
  O15["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- fine_tuned_readout_power<br/>- fine_tuned_readout_frequency</div>"]
  I15 --> E15 --> O15
end

subgraph Exp16[" "]
  direction LR
  I16["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- (measurement only)</div>"]
  E16["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>IQ Blob</div>"]
  O16["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- IQ_rotation_angle<br/>- discrimination_threshold</div>"]
  I16 --> E16 --> O16
end

subgraph Exp17[" "]
  direction LR
  I17["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- wait_time<br/>- qubit_flux</div>"]
  E17["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Ramsey<br/>vs Qubit Flux</div>"]
  O17["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- joint_UPSS_voltages_set</div>"]
  I17 --> E17 --> O17
end

subgraph Exp18[" "]
  direction LR
  I18["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- coupler_flux</div>"]
  E18["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>ZZ off<br/>JAZZ</div>"]
  O18["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- joint_UPSS_coupler_voltage_set</div>"]
  I18 --> E18 --> O18
end


subgraph Exp19[" "]
  direction LR
  I19["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- wait_time</div>"]
  E19["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>T1</div>"]
  O19["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- T1</div>"]
  I19 --> E19 --> O19
end

subgraph Exp20[" "]
  direction LR
  I20["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- wait_time<br/>- detuning_signs<br/></div>"]
  E20["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>T2*</div>"]
  O20["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- T2*</div>"]
  I20 --> E20 --> O20
end

subgraph Exp21[" "]
  direction LR
  I21["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- num_random_sequence<br/>- delta_clifford<br/>- circuit_depth</div>"]
  E21["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>1Q Randomized<br/>Benchmarking</div>"]
  O21["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- one_qubit_fidelity</div>"]
  I21 --> E21 --> O21
end

subgraph Exp22[" "]
  direction LR
  I22["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- drive_frequency<br/>- qubit_flux_duration</div>"]
  E22["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Cryoscope<br/>(IIR for Qubit Flux)</div>"]
  O22["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- digital_filter_IIR</div>"]
  I22 --> E22 --> O22
end

subgraph Exp23[" "]
  direction LR
  I23["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>-frame_rotation<br/>- qubit_flux_duration</div>"]
  E23["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Cryoscope<br/>(FIR for Qubit Flux)</div>"]
  O23["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- digital_filter_FIR</div>"]
  I23 --> E23 --> O23
end

subgraph Exp24[" "]
  direction LR
  I24["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>-qubit_frequency<br/>- coupler_flux_duration</div>"]
  E24["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Cryoscope<br/>(IIR for Coupler Flux)</div>"]
  O24["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- digital_filter_IIR</div>"]
  I24 --> E24 --> O24
end

subgraph Exp25[" "]
  direction LR
  I25["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>-frame_rotation<br/>- coupler_flux_duration</div>"]
  E25["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Cryoscope<br/>(FIR for Coupler Flux)</div>"]
  O25["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- digital_filter_FIR</div>"]
  I25 --> E25 --> O25
end

subgraph Exp26[" "]
  direction LR
  I26["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- relative_delay</div>"]
  E26["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>XY-Z Delay</div>"]
  O26["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- Z_delay</div>"]
  I26 --> E26 --> O26
end

subgraph Exp27[" "]
  direction LR
  I27["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- relative_delay</div>"]
  E27["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>XY-Coupler Delay</div>"]
  O27["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- coupler_delay</div>"]
  I27 --> E27 --> O27
end

subgraph Exp28[" "]
  direction LR
  I28["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- ef_frequency</div>"]
  E28["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>EF Spectroscopy</div>"]
  O28["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- anharmonicity</div>"]
  I28 --> E28 --> O28
end

subgraph Exp29[" "]
  direction LR
  I29["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- readout_amplitude</div>"]
  E29["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>EF Rabi and Readout Opt</div>"]
  O29["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- ef_pi_amp<br/>- gef_ro_freq</div>"]
  I29 --> E29 --> O29
end

subgraph Exp30[" "]
  direction LR
  I30["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- readout_frequency</div>"]
  E30["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>GEF Readout<br/> Optimization</div>"]
  O30["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- ef_pi_amp<br/>- gef_ro_freq</div>"]
  I30 --> E30 --> O30
end

subgraph Exp31[" "]
  direction LR
  I31["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- flux_amplitude<br/>- gate_time</div>"]
  E31["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>02-11 leakage<br/> & condi-Z</div>"]
  O31["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- qubit_flux_amp<br/>- coupler_flux_amp</div>"]
  I31 --> E31 --> O31
end

subgraph Exp32[" "]
  direction LR
  I32["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- qubit_flux_amplitude</div>"]
  E32["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>1D cond_Z<br/> vs Qubit Flux Bias</div>"]
  O32["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- qubit_flux_amp</div>"]
  I32 --> E32 --> O32
end

subgraph Exp33[" "]
  direction LR
  I33["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- qubit_flux_amplitude</div>"]
  E33["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Error Amplification<br/> 1D cond_Z<br/> vs Qubit Flux Bias</div>"]
  O33["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- qubit_flux_amp</div>"]
  I33 --> E33 --> O33
end

subgraph Exp34[" "]
  direction LR
  I34["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- coupler_flux_amplitude</div>"]
  E34["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Error Amplification<br/> of leakage</div>"]
  O34["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- coupler_flux_amp</div>"]
  I34 --> E34 --> O34
end

subgraph Exp35[" "]
  direction LR
  I35["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- frame_rotation</div>"]
  E35["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Trivial Phase<br/> Compensation</div>"]
  O35["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- phase_cor_qij</div>"]
  I35 --> E35 --> O35
end

subgraph Exp36[" "]
  direction LR
  I36["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- (measurement only)</div>"]
  E36["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Bell State<br/> Tomography</div>"]
  O36["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- entangled_fid</div>"]
  I36 --> E36 --> O36
end

subgraph Exp37[" "]
  direction LR
  I37["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- circuit_depth</div>"]
  E37["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>Two-Qubit<br/> Randomized Benchmarking</div>"]
  O37["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- 2q_fid</div>"]
  I37 --> E37 --> O37
end

subgraph Exp38[" "]
  direction LR
  I38["<div style='width:420px; text-align:center; font-size:16px'><b>Node params</b><br/>- circuit_depth</div>"]
  E38["<div style='width:320px; text-align:center; font-size:20px; font-weight:700'>XEB</div>"]
  O38["<div style='width:420px; text-align:center; font-size:16px'><b>Output</b><br/>- 2q_fid</div>"]
  I38 --> E38 --> O38
end

%% Vertical workflow links
Exp1 --> Exp2 --> Exp3 --> Exp4 --> Exp5 --> Exp6 --> Exp7 --> Exp8 --> Exp9 --> Exp10 --> Exp11 --> Exp12 --> Exp13 --> Exp14 --> Exp15 --> Exp16 --> Exp17 --> Exp18 --> Exp19 --> Exp20 --> Exp21 --> Exp22 --> Exp23 --> Exp24 --> Exp25 --> Exp26 --> Exp27 --> Exp28 --> Exp29 --> Exp30 --> Exp31 --> Exp32 --> Exp33 --> Exp34 --> Exp35 --> Exp36 --> Exp37 --> Exp38
```
