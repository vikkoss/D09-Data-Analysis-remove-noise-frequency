%% DATA FILES %%
In this zip file you will find a total of 36 *.mat files: one for each pilot (6) 
and condition (6) combination. Pilots are numbered 1-6. Conditions are numbered
C1-C6, with the following order:

  - C1 = Gain (P), no motion
  - C2 = Single integrator (V), no motion
  - C3 = Double integrator (A), no motion
  - C4 = Gain (P), motion
  - C5 = Single integrator (V), motion
  - C6 = Double integrator (A), motion

In each *.mat data files you will find:

%% TIME HISTORIES %%
	- the error signal              e [deg]
	- the control signal            u [deg]	
	- the controlled yaw angle      x	[deg]
  - the target signal            ft [deg]
  - the disturbance signal       fd [deg]
  - the time vector               t [s]

%% MEASURED PILOT FREQUENCY RESPONSES %%
  - the Hpe (visual) frequency response   Hpe_FC [complex numbers]
  - the Hpxd (motion) frequency response Hpxd_FC [complex numbers]
  - the frequency vector                    w_FC [rad/s]    


Good luck with the data analysis! 
