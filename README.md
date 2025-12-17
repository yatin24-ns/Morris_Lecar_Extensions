# Morris_Lecar_Extensions
A temporary landing site for all deliverables related to the project "From Spikes to Bursts — Modeling Calcium-Mediated Neural Dynamics with Stochastic 3D Morris–Lecar Extensions"

An interactive, Python-based simulator for exploring the dynamics of the Morris-Lecar neuron model. This tool is designed for students and researchers to visualize phase plane geometry, bifurcation theory, and biological excitability.

Features
Real-Time Phase Plane Analysis: visualize Nullclines, Vector Fields, and Fixed Point stability (Stable, Unstable, Saddle) in real-time.
Live Simulation: Click anywhere on the phase plane to launch a trajectory, or inject current pulses to trigger action potentials.
Biologically Accurate: Uses standard parameter sets (Rinzel & Ermentrout, 1989) for Class I (SNIC) and Class II (Hopf) excitability.
Transition Analysis: Includes a "Bogdanov-Takens" mode to explore the geometric transition between neuron classes.

Installation
Clone this repository or download the files.
Install the required Python libraries using pip:
Run the main script from your terminal:
python classical.py

You will be prompted to select a neuron model:
Class II (Squid Axon): Fast onset, discontinuous frequency. Best for beginners.
Class I (Mollusk): Slow onset, continuous frequency.
Bogdanov-Takens: Advanced mode for exploring bifurcations.

Controls
Current Slider: Inject constant current to depolarize the cell.
Pulse Mag: Set the strength of the momentary stimulus.
Spike (+): Inject a positive current pulse to trigger a spike.
Rebound (-): Inject a negative current pulse to demonstrate Anode Break Excitation.
V3 Shift: (Transition Mode Only) Shift the nullcline to change the bifurcation topology.

License:
To the extent possible under law, the author has waived all copyrights and related or neighboring rights to this work.
