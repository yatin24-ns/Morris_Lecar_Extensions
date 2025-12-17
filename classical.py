"""
Morris-Lecar Neuron Simulator (Biology Project)
A simulation of neural excitability using the Morris-Lecar model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from scipy.optimize import brentq
from scipy.signal import find_peaks
import sys

# --- 1. BIOLOGICAL CONSTANTS ---
C_m = 20.0      # Membrane Capacitance (uF/cm^2)
V_Ca = 120.0    # Ca2+ Nernst Potential (mV)
V_K = -84.0     # K+ Nernst Potential (mV)
V_L = -60.0     # Leak Potential (mV)
g_L = 2.0       # Leak Conductance (mS/cm^2)

# --- 2. MODEL PARAMETERS ---
MODELS = {
    'Class I (Mollusk)': {
        'name': "Class I (Mollusk/Snail)",
        'desc': "Slow spiking. Frequency can be arbitrarily low.",
        'g_Ca': 4.0, 'g_K': 8.0, 
        'v1': -1.2, 'v2': 18.0, 'v3': 12.0, 'v4': 17.4, 'phi': 0.0667,
        'I_max': 150.0, 'show_v3_slider': False
    },
    'Class II (Squid)': {
        'name': "Class II (Squid Axon)",
        'desc': "Fast spiking. Firing starts immediately at high frequency.",
        'g_Ca': 4.4, 'g_K': 8.0, 
        'v1': -1.2, 'v2': 18.0, 'v3': 2.0, 'v4': 30.0, 'phi': 0.04,
        'I_max': 150.0, 'show_v3_slider': False
    },
    'Transition': {
        'name': "Transition (Theoretical)",
        'desc': "Observe how shifting K+ kinetics changes the bifurcation.",
        'g_Ca': 4.0, 'g_K': 8.0, 
        'v1': -1.2, 'v2': 18.0, 'v3': 12.0, 'v4': 17.4, 'phi': 0.0667,
        'I_max': 100.0, 'show_v3_slider': True
    }
}

current_model = MODELS['Class I (Mollusk)'] # Default to Class I
p = current_model.copy() # Load parameters

# --- 3. BIOPHYSICAL EQUATIONS ---

def m_inf(V): 
    return 0.5 * (1 + np.tanh((V - p['v1']) / p['v2']))

def w_inf(V, v3_val): 
    return 0.5 * (1 + np.tanh((V - v3_val) / p['v4']))

def tau_w(V, v3_val): 
    return 1.0 / np.cosh((V - v3_val) / (2 * p['v4']))

def neural_derivatives(state, t, applied_current, v3_val, pulse=0):
    V, w = state
    total_current = applied_current + pulse
    
    I_Ca = p['g_Ca'] * m_inf(V) * (V - V_Ca)
    I_K = p['g_K'] * w * (V - V_K)
    I_L = g_L * (V - V_L)
    
    dVdt = (total_current - I_Ca - I_K - I_L) / C_m
    dwdt = p['phi'] * (w_inf(V, v3_val) - w) / tau_w(V, v3_val)
    
    return [dVdt, dwdt]

# --- 4. ANALYSIS TOOLS ---

def calculate_nullclines(voltage_range, I_app, v3_val):
    denominator = p['g_K'] * (voltage_range - V_K) + 1e-9 
    numerator = I_app - p['g_Ca'] * m_inf(voltage_range) * (voltage_range - V_Ca) - g_L * (voltage_range - V_L)
    v_nullcline = numerator / denominator
    w_nullcline = w_inf(voltage_range, v3_val)
    return v_nullcline, w_nullcline

def find_equilibrium_points(I_app, v3_val):
    def net_current(V):
        w = w_inf(V, v3_val)
        return I_app - p['g_Ca']*m_inf(V)*(V-V_Ca) - p['g_K']*w*(V-V_K) - g_L*(V-V_L)

    V_scan = np.linspace(-100, 100, 600)
    signs = np.sign(net_current(V_scan))
    roots = []
    
    if len(signs) > 0:
        for i in np.where(np.diff(signs))[0]:
            try:
                root = brentq(net_current, V_scan[i], V_scan[i+1])
                roots.append(root)
            except: pass
    
    points = {'stable': [], 'unstable': [], 'saddle': []}
    for r in roots:
        w_val = w_inf(r, v3_val)
        # Jacobian calc for stability
        dm = (0.5/p['v2']) * (1 - np.tanh((r-p['v1'])/p['v2'])**2)
        dw = (0.5/p['v4']) * (1 - np.tanh((r-v3_val)/p['v4'])**2)
        j11 = (-p['g_Ca']*(dm*(r-V_Ca)+m_inf(r)) - p['g_K']*w_val - g_L)/C_m
        j12 = -p['g_K']*(r-V_K)/C_m
        j21 = p['phi'] * dw / tau_w(r, v3_val)
        j22 = -p['phi'] / tau_w(r, v3_val)
        
        tr = j11 + j22
        det = j11*j22 - j12*j21
        
        if det < 0: points['saddle'].append((r, w_val))
        elif tr < 0: points['stable'].append((r, w_val))
        else: points['unstable'].append((r, w_val))
            
    return points

# --- 5. GRAPHICS SETUP ---

# Globals
active_animations = []
stored_lines = []
stored_arrows = []
stored_fills = []  # NEW: For storing limit cycle shading
quiver_dense = None
quiver_sparse = None

fig = plt.figure(figsize=(16, 9), facecolor='#fafafa')
grid = fig.add_gridspec(3, 2, width_ratios=[1.4, 1], height_ratios=[1, 1, 0.6])
# INCREASED BOTTOM MARGIN to 0.30
# INCREASED wspace to 0.3 to prevent y-axis label overlap
plt.subplots_adjust(left=0.06, bottom=0.30, right=0.98, top=0.95, wspace=0.3, hspace=0.35)

ax_phase = fig.add_subplot(grid[:, 0])
ax_volt = fig.add_subplot(grid[0, 1])
ax_curr = fig.add_subplot(grid[1, 1])
ax_info = fig.add_subplot(grid[2, 1])
ax_info.axis('off')

# Phase Plane
V_range = np.linspace(-80, 60, 500)
line_wnc, = ax_phase.plot([], [], 'g-', lw=2.5, alpha=0.8, label='w-nullcline (K+)')
line_vnc, = ax_phase.plot([], [], 'b-', lw=2.5, alpha=0.8, label='V-nullcline (Balance)')

# Markers
markers = {
    'stable': ax_phase.plot([], [], 'go', ms=10, label='Stable')[0],
    'unstable': ax_phase.plot([], [], 'ro', ms=10, label='Unstable')[0],
    'saddle': ax_phase.plot([], [], 'x', color='#8B0000', ms=12, markeredgewidth=3, label='Saddle')[0]
}

# Vector Fields
grid_V_d, grid_w_d = np.meshgrid(np.linspace(-75, 50, 30), np.linspace(-0.1, 0.7, 30))
quiver_dense = ax_phase.quiver(grid_V_d, grid_w_d, np.zeros_like(grid_V_d), np.zeros_like(grid_V_d), 
                              color='gray', alpha=0.3, pivot='mid')
grid_V_s, grid_w_s = np.meshgrid(np.linspace(-75, 50, 20), np.linspace(-0.1, 0.7, 20))
quiver_sparse = ax_phase.quiver(grid_V_s, grid_w_s, np.zeros_like(grid_V_s), np.zeros_like(grid_V_s), 
                               color='black', alpha=0.4, pivot='mid', width=0.0025, headwidth=5, headlength=5)
quiver_sparse.set_visible(False)

ax_phase.set_title("Phase Plane Analysis", fontsize=12, fontweight='bold')
ax_phase.set_xlabel("Voltage V (mV)")
ax_phase.set_ylabel("K+ Recovery w")
ax_phase.set_xlim(-75, 50); ax_phase.set_ylim(-0.1, 0.7)
ax_phase.grid(True, alpha=0.3)
ax_phase.minorticks_on()
ax_phase.legend(loc='upper left', fontsize=8, fancybox=True)

# Voltage Graph
ax_volt.set_title("Neuron Output", fontsize=12, fontweight='bold')
ax_volt.set_ylabel("Voltage (mV)"); ax_volt.set_xlim(0, 500); ax_volt.set_ylim(-80, 50)
ax_volt.grid(True, alpha=0.3)
ax_volt.minorticks_on()
line_rest, = ax_volt.plot([], [], '--', color='#FFD700', lw=2, label='Resting V')
ax_volt.legend(loc='upper right', fontsize=8)
text_status = ax_volt.text(0.03, 0.78, "State: Ready", transform=ax_volt.transAxes, 
                           fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

# Current Graph
ax_curr.set_title("Ionic Currents", fontsize=12, fontweight='bold')
ax_curr.set_ylabel("uA/cm^2"); ax_curr.set_xlabel("Time (ms)")
ax_curr.set_xlim(0, 500); ax_curr.set_ylim(-300, 300)
ax_curr.grid(True, alpha=0.3); ax_curr.axhline(0, color='black', lw=0.5)
ax_curr.minorticks_on()
ax_curr.plot([], [], 'r-', label='Calcium (In)')
ax_curr.plot([], [], 'b-', label='Potassium (Out)')
ax_curr.legend(loc='upper right', fontsize=9)

# Info Box
text_results = ax_info.text(0.5, 0.5, "", ha='center', va='center', wrap=True,
                            bbox=dict(boxstyle="round,pad=1", fc="white", ec="purple"))

# --- 6. CONTROLS ---

# Radio Buttons (Far Left) - Shifted Left
ax_radio_model = plt.axes([0.02, 0.05, 0.15, 0.20], facecolor='#f0f0f0')
radio_model = RadioButtons(ax_radio_model, list(MODELS.keys()))

# Vector Toggle (Next to Model) - Shifted Left
ax_radio_vec = plt.axes([0.18, 0.05, 0.10, 0.10], facecolor='#f0f0f0')
radio_vec = RadioButtons(ax_radio_vec, ['Dense', 'Sparse'])

# Sliders (Center-Right, Stacked)
# Shifted x start to 0.42 to clear radio buttons & provide space for labels
# Reduced width to 0.32 to fit before buttons
# Row 1 (Top): Current
ax_slider_I = plt.axes([0.42, 0.18, 0.32, 0.03], facecolor='lightgoldenrodyellow')
slider_I = Slider(ax_slider_I, 'Current', 0, 150, valinit=0)

# Row 2 (Mid): V3 Shift
ax_slider_v3 = plt.axes([0.42, 0.13, 0.32, 0.03], facecolor='#e6e6fa')
slider_v3 = Slider(ax_slider_v3, 'V3 Shift', 0, 15, valinit=2)

# Row 3 (Bot): Pulse Mag
ax_slider_pulse = plt.axes([0.42, 0.08, 0.32, 0.03], facecolor='#ffcccc')
slider_pulse = Slider(ax_slider_pulse, 'Pulse Mag', 0, 100, valinit=40)

# Action Buttons (Far Right Column) - Shifted slightly right
btn_spike = Button(plt.axes([0.78, 0.18, 0.08, 0.05]), 'Spike (+)', color='#ffcccc')
btn_rebound = Button(plt.axes([0.88, 0.18, 0.10, 0.05]), 'Rebound (-)', color='#ccccff')
btn_clear = Button(plt.axes([0.78, 0.08, 0.20, 0.05]), 'Clear All', color='white')

# --- 7. LOGIC ---

# KEYBOARD CONTROL LOGIC
sliders = [slider_I, slider_v3, slider_pulse]
active_slider_idx = 0

def update_slider_visuals():
    """Highlights the label of the active slider."""
    for i, s in enumerate(sliders):
        if i == active_slider_idx:
            s.label.set_color('red')
            s.label.set_fontweight('bold')
        else:
            s.label.set_color('black')
            s.label.set_fontweight('normal')
    fig.canvas.draw_idle()

def on_key_press(event):
    global active_slider_idx
    
    # Selection: UP/DOWN cycles through sliders
    if event.key == 'up':
        active_slider_idx = (active_slider_idx - 1) % len(sliders)
        update_slider_visuals()
    elif event.key == 'down':
        active_slider_idx = (active_slider_idx + 1) % len(sliders)
        update_slider_visuals()
    
    # Adjustment: LEFT/RIGHT changes value
    elif event.key in ['left', 'right']:
        s = sliders[active_slider_idx]
        # Skip if slider is hidden (e.g. V3 shift in certain models)
        if not s.ax.get_visible():
            return
            
        step = (s.valmax - s.valmin) * 0.02 # 2% step
        if event.key == 'left':
            new_val = s.val - step
        else:
            new_val = s.val + step
        
        # Clamp and set
        new_val = max(s.valmin, min(s.valmax, new_val))
        s.set_val(new_val)

fig.canvas.mpl_connect('key_press_event', on_key_press)
update_slider_visuals()

def update_vector_field(I, v3):
    mode = radio_vec.value_selected
    
    # Reset
    quiver_dense.set_visible(False)
    quiver_sparse.set_visible(False)
    
    # Choose Grid
    if mode == 'Dense':
        grid_V, grid_w = grid_V_d, grid_w_d
        quiver_active = quiver_dense
        quiver_dense.set_visible(True)
    else:
        # 'Sparse' or 'Separatrix' uses the sparse grid
        grid_V, grid_w = grid_V_s, grid_w_s
        quiver_active = quiver_sparse
        quiver_sparse.set_visible(True)

    # Calc derivatives
    i_ca = p['g_Ca'] * m_inf(grid_V) * (grid_V - V_Ca)
    i_k = p['g_K'] * grid_w * (grid_V - V_K)
    i_l = g_L * (grid_V - V_L)
    dv = (I - i_ca - i_k - i_l) / C_m
    dw = p['phi'] * (w_inf(grid_V, v3) - grid_w) / tau_w(grid_V, v3)
    
    # Aspect Ratio Correction
    aspect = (50 - (-75)) / (0.7 - (-0.1))
    dw_vis = dw * aspect
    mag = np.sqrt(dv**2 + dw_vis**2)
    mag[mag==0] = 1
    
    quiver_active.set_UVC(dv/mag, dw_vis/mag)

def update_plot(val=None):
    I = slider_I.val
    v3 = slider_v3.val if slider_v3.ax.get_visible() else p['v3']
    
    vn, wn = calculate_nullclines(V_range, I, v3)
    line_vnc.set_data(V_range, vn)
    line_wnc.set_data(V_range, wn)
    
    pts = find_equilibrium_points(I, v3)
    for k, mk in markers.items():
        if pts[k]: mk.set_data(*zip(*pts[k]))
        else: mk.set_data([], [])
            
    if pts['stable']:
        r = pts['stable'][0][0]
        line_rest.set_data([0, 1000], [r, r])
    else:
        line_rest.set_data([], [])

    update_vector_field(I, v3)
    
    text_results.set_text(current_model['desc'])
    fig.canvas.draw_idle()

def change_model(label):
    global current_model, p
    current_model = MODELS[label]
    p.update(current_model)
    slider_I.valmax = current_model['I_max']
    slider_I.set_val(0) 
    slider_v3.set_val(p['v3']) 
    slider_v3.ax.set_visible(current_model['show_v3_slider'])
    clear_all(None)
    update_plot()

def toggle_vectors(label):
    update_plot()

def run_simulation(start_V, start_w, pulse_mag=0):
    t = np.linspace(0, 500, 1500)
    I_base = slider_I.val
    v3 = slider_v3.val if slider_v3.ax.get_visible() else p['v3']
    
    if pulse_mag != 0:
        s1 = odeint(neural_derivatives, [start_V, start_w], t[:150], args=(I_base, v3, 0))
        s2 = odeint(neural_derivatives, s1[-1], t[:150], args=(I_base, v3, pulse_mag))
        s3 = odeint(neural_derivatives, s2[-1], t[300:], args=(I_base, v3, 0))
        sol = np.vstack([s1, s2, s3])
    else:
        sol = odeint(neural_derivatives, [start_V, start_w], t, args=(I_base, v3, 0))
        
    v_t = sol[:,0]; w_t = sol[:,1]
    i_ca = p['g_Ca'] * m_inf(v_t) * (v_t - V_Ca)
    i_k = p['g_K'] * w_t * (v_t - V_K)
    
    l_phase, = ax_phase.plot([], [], 'k-', lw=1.0, alpha=0.5)
    l_volt, = ax_volt.plot([], [], 'k-', lw=1.0, alpha=0.5)
    l_ca, = ax_curr.plot([], [], 'r-', lw=1.0, alpha=0.5)
    l_k, = ax_curr.plot([], [], 'b-', lw=1.0, alpha=0.5)
    
    stored_lines.extend([l_phase, l_volt, l_ca, l_k])
    
    peaks, _ = find_peaks(v_t, height=-20)
    n = len(peaks)
    
    # --- LIMIT CYCLE SHADING LOGIC ---
    if n >= 2:
        # Extract the last full cycle (from 2nd last peak to last peak)
        p_idx = peaks
        start_idx = p_idx[-2]
        end_idx = p_idx[-1]
        
        cycle_V = v_t[start_idx:end_idx+1]
        cycle_w = w_t[start_idx:end_idx+1]
        
        # Shade the area inside the limit cycle
        poly = ax_phase.fill(cycle_V, cycle_w, color='red', alpha=0.15, zorder=0)
        stored_fills.extend(poly)
    # ---------------------------------

    if n == 0:
        status = "State: Resting"; col = "green"
        res = "RESULT: Subthreshold. Stimulus too weak."
    elif n == 1:
        status = "Event: Spike"; col = "orange"
        res = f"RESULT: Single {'Rebound ' if pulse_mag<0 else ''}Spike."
    else:
        freq = n / 0.5
        status = f"Spiking: {freq:.1f} Hz"; col = "red"
        res = f"RESULT: Repetitive Spiking at {freq:.1f} Hz."
        
    text_results.set_text(current_model['desc'] + "\n" + "-"*30 + "\n" + res)
    
    active_animations.append({
        'lines': [l_phase, l_volt, l_ca, l_k],
        'data': [sol, t, i_ca, i_k],
        'idx': 0, 'status': (status, col)
    })

def update_animation(frame):
    for anim in active_animations[::-1]:
        idx = anim['idx']
        nxt = min(idx + 30, len(anim['data'][0]))
        sol, t, ca, k = anim['data']
        
        anim['lines'][0].set_data(sol[:nxt, 0], sol[:nxt, 1])
        anim['lines'][1].set_data(t[:nxt], sol[:nxt, 0])
        anim['lines'][2].set_data(t[:nxt], ca[:nxt])
        anim['lines'][3].set_data(t[:nxt], k[:nxt])
        
        txt, col = anim['status']
        text_status.set_text(txt); text_status.set_color(col)
        
        if nxt % 10 == 0 and nxt < len(sol):
            dist = np.linalg.norm(sol[nxt] - sol[idx])
            if dist > 0.5:
                arr = ax_phase.annotate('', xy=sol[nxt], xytext=sol[nxt-3],
                                        arrowprops=dict(arrowstyle="->", lw=2, color='red'))
                stored_arrows.append(arr)
        
        anim['idx'] = nxt
        if nxt >= len(sol): active_animations.remove(anim)

def clear_all(event):
    for l in stored_lines: l.remove()
    stored_lines.clear()
    for a in stored_arrows: a.remove()
    stored_arrows.clear()
    for f in stored_fills: f.remove() # Remove shading
    stored_fills.clear()
    for anim in active_animations:
        for l in anim['lines']: 
            if l is not None: l.remove()
    active_animations.clear()
    text_status.set_text("State: Ready"); text_status.set_color("black")
    text_results.set_text(current_model['desc'])
    fig.canvas.draw_idle()

def main():
    update_plot()
    
    slider_I.on_changed(update_plot)
    slider_v3.on_changed(update_plot)
    radio_model.on_clicked(change_model)
    radio_vec.on_clicked(toggle_vectors)
    
    def get_start_point():
        v3_val = slider_v3.val if slider_v3.ax.get_visible() else p['v3']
        eq = find_equilibrium_points(slider_I.val, v3_val)
        if eq['stable']: return eq['stable'][0]
        return (-60, 0.01)

    btn_spike.on_clicked(lambda x: run_simulation(*get_start_point(), pulse_mag=slider_pulse.val))
    btn_rebound.on_clicked(lambda x: run_simulation(*get_start_point(), pulse_mag=-slider_pulse.val))
    btn_clear.on_clicked(clear_all)
    fig.canvas.mpl_connect('button_press_event', lambda e: run_simulation(e.xdata, e.ydata) if e.inaxes==ax_phase else None)
    
    plt.show()

ani = FuncAnimation(fig, update_animation, interval=30, cache_frame_data=False)

if __name__ == "__main__":
    main()
