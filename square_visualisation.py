#We will use similar setups for all bursting sims (probably, might change depending on the specifics.)
# VISUALIZATION SETUP
fig = plt.figure(figsize=(15, 9), facecolor='#fafafa')
# --- Layout Configuration ---
# 4 Rows on Right, 1 Full Height Column on Left
grid = fig.add_gridspec(4, 2, width_ratios=[1.8, 1], height_ratios=[1.2, 1, 0.8, 1.2])
plt.subplots_adjust(left=0.06, bottom=0.25, right=0.96, top=0.93, wspace=0.15, hspace=0.5)

# --- LEFT COLUMN: MAIN VIEWS (Stacked Axes) ---
# VIEW 1: 2D Slow Phase Plane (Active by default)
ax_phase_2d = fig.add_subplot(grid[:, 0])
ax_phase_2d.set_title("1/3: Slow Manifold Projection (V vs z)", fontsize=12, fontweight='bold')
ax_phase_2d.set_xlabel("Slow Variable ($z$)")
ax_phase_2d.set_ylabel("Voltage (V)")
ax_phase_2d.set_xlim(-0.1, 1.1)
ax_phase_2d.set_ylim(-75, 50)
ax_phase_2d.grid(True, alpha=0.3)

# Static/Dynamic Elements for View 1
line_equil, = ax_phase_2d.plot([], [], color='k', lw=3, alpha=0.6, label='Nullcline')
line_traj_2d, = ax_phase_2d.plot([], [], 'c-', lw=2, alpha=0.9, label='Trajectory')
point_curr_2d, = ax_phase_2d.plot([], [], 'o', color='magenta', ms=10, markeredgecolor='white')

# Bifurcation Markers
point_sn, = ax_phase_2d.plot([], [], '*', color='gold', ms=12, markeredgecolor='black', label='SN (Start)')
point_hopf, = ax_phase_2d.plot([], [], 'X', color='red', ms=10, markeredgecolor='black', label='Hopf (End)')

# Dynamic Threshold Lines (Gold Dashed)
line_thresh_sn, = ax_phase_2d.plot([], [], color='gold', linestyle='--', alpha=0.7)
line_thresh_hopf, = ax_phase_2d.plot([], [], color='gold', linestyle='--', alpha=0.7)

# Phase Labels (The "Resting and Spiking thing" on Phase Plot)
ax_phase_2d.text(0.05, -65, "SILENT PHASE\n(Charging)", color='blue', fontsize=11, fontweight='bold', ha='left')
ax_phase_2d.text(0.70, 40, "ACTIVE PHASE\n(Bursting)", color='red', fontsize=11, fontweight='bold', ha='left')

# Line Labels
text_sn_label = ax_phase_2d.text(0.02, 0, "Saddle-Node", color='goldenrod', fontsize=9, fontweight='bold')
text_hopf_label = ax_phase_2d.text(0.02, 0, "Hopf/Homoclinic", color='goldenrod', fontsize=9, fontweight='bold')
ax_phase_2d.legend(loc='upper right', fontsize=9)

# VIEW 2: 3D Phase Space (Hidden by default)
ax_phase_3d = fig.add_subplot(grid[:, 0], projection='3d')
ax_phase_3d.set_title("2/3: 3D Phase Space (V, w, z)", fontsize=12, fontweight='bold')
ax_phase_3d.set_xlabel('z (Slow)')
ax_phase_3d.set_ylabel('w (Recovery)')
ax_phase_3d.set_zlabel('V (Voltage)')
ax_phase_3d.set_xlim(0, 1.0)
ax_phase_3d.set_ylim(0, 0.6)
ax_phase_3d.set_zlim(-75, 50)
ax_phase_3d.view_init(elev=20, azim=-45)
ax_phase_3d.set_visible(False)

line_traj_3d, = ax_phase_3d.plot([], [], [], 'r-', lw=1, alpha=0.7)
point_curr_3d, = ax_phase_3d.plot([], [], [], 'o', color='magenta', ms=8)

# VIEW 3: 2D Fast Phase Plane (Hidden by default)
ax_phase_fast = fig.add_subplot(grid[:, 0])
ax_phase_fast.set_title("3/3: Fast Subsystem (V vs w)", fontsize=12, fontweight='bold')
ax_phase_fast.set_xlabel("Voltage (V)")
ax_phase_fast.set_ylabel("Recovery (w)")
ax_phase_fast.set_xlim(-80, 50)
ax_phase_fast.set_ylim(0, 0.7)
ax_phase_fast.grid(True, alpha=0.3)
ax_phase_fast.set_visible(False)

v_range_fast = np.linspace(-80, 50, 400)
ax_phase_fast.plot(v_range_fast, w_inf(v_range_fast), 'g--', lw=2, alpha=0.6, label='$w_{\infty}$')
line_v_null_fast, = ax_phase_fast.plot([], [], 'orange', lw=3, label='$V$-null')
line_traj_fast, = ax_phase_fast.plot([], [], 'purple', lw=2, alpha=0.8)
point_curr_fast, = ax_phase_fast.plot([], [], 'o', color='magenta', ms=10)
ax_phase_fast.legend(loc='upper left', fontsize=9)

# --- RIGHT COLUMN: ANALYSIS ---
# 1. Voltage Trace
ax_volt = fig.add_subplot(grid[0, 1])
ax_volt.set_title("Voltage Trace (V)", fontsize=10, fontweight='bold')
ax_volt.set_ylabel("V (mV)")
ax_volt.set_ylim(-75, 50)
ax_volt.set_xlim(0, 10000)
ax_volt.grid(True, alpha=0.3)
line_volt, = ax_volt.plot([], [], 'k-', lw=0.8)

# Status Box (The "thing above voltage plot")
text_status_box = ax_volt.text(0.02, 0.85, "SILENT (Recovery)", 
                               transform=ax_volt.transAxes, 
                               fontsize=10, fontweight='bold', color='blue',
                               bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

# 2. Slow Variable Trace
ax_slow = fig.add_subplot(grid[1, 1])
ax_slow.set_title("Slow Variable (z)", fontsize=10, fontweight='bold')
ax_slow.set_ylabel("z")
ax_slow.set_ylim(0, 1.0)
ax_slow.set_xlim(0, 10000)
ax_slow.grid(True, alpha=0.3)
line_slow, = ax_slow.plot([], [], 'b-', lw=1.0)

# 3. Noise Trace
ax_noise = fig.add_subplot(grid[2, 1])
ax_noise.set_title("Stochastic Noise", fontsize=10, fontweight='bold')
ax_noise.set_ylabel("Noise")
ax_noise.set_ylim(-3, 3)
ax_noise.set_xlim(0, 10000)
ax_noise.grid(True, alpha=0.3)
line_noise, = ax_noise.plot([], [], 'g-', lw=0.5, alpha=0.6)

# 4. IBI Histogram & CV
ax_ibi = fig.add_subplot(grid[3, 1])
ax_ibi.set_title("IBI Distribution", fontsize=10, fontweight='bold')
ax_ibi.set_ylabel("Count")
ax_ibi.set_xlabel("Interval (ms)")
ax_ibi.grid(True, alpha=0.3)
# Placeholder text for CV
text_cv = ax_ibi.text(0.95, 0.85, "Waiting for bursts...", 
                      transform=ax_ibi.transAxes, ha='right', va='center',
                      fontsize=10, fontweight='bold', color='grey', 
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


# --- CONTROLS ---
# View Toggle Button
ax_btn_view = plt.axes([0.06, 0.93, 0.12, 0.04]) 
btn_view = Button(ax_btn_view, 'Cycle Views', color='lightblue', hovercolor='skyblue')

# Sliders (Stacked Logic)
ax_sNoise = plt.axes([0.15, 0.12, 0.25, 0.03], facecolor='salmon')
ax_sEps   = plt.axes([0.55, 0.12, 0.25, 0.03], facecolor='lavender')
ax_sI     = plt.axes([0.15, 0.05, 0.25, 0.03], facecolor='lightgoldenrodyellow')
btn_reset_ax = plt.axes([0.65, 0.05, 0.1, 0.04])

s_Noise = Slider(ax_sNoise, 'Noise', 0.0, 0.2, valinit=noise_default)
s_Eps   = Slider(ax_sEps, 'Speed (z)', 0.001, 0.02, valinit=epsilon_default)
s_I     = Slider(ax_sI, 'Base Current', 35.0, 55.0, valinit=I_app_default)
btn_reset = Button(btn_reset_ax, 'Reset', color='white', hovercolor='grey')

# --- LOGIC HANDLERS ---
# View Cycle Logic
view_mode = '2D_Slow' # '2D_Slow', '3D', '2D_Fast'

def toggle_view(event):
    global view_mode
    # Hide all
    ax_phase_2d.set_visible(False)
    ax_phase_3d.set_visible(False)
    ax_phase_fast.set_visible(False)

    if view_mode == '2D_Slow':
        view_mode = '3D'
        ax_phase_3d.set_visible(True)
    elif view_mode == '3D':
        view_mode = '2D_Fast'
        ax_phase_fast.set_visible(True)
    else:
        view_mode = '2D_Slow'
        ax_phase_2d.set_visible(True)
    
    fig.canvas.draw_idle()

btn_view.on_clicked(toggle_view)

# Keyboard Logic
sliders = [s_Noise, s_I, s_Eps]
active_slider_idx = 1 
def update_slider_visuals():
    for i, s in enumerate(sliders):
        s.label.set_color('red' if i == active_slider_idx else 'black')
        s.label.set_fontweight('bold' if i == active_slider_idx else 'normal')
    fig.canvas.draw_idle()

def on_key_press(event):
    global active_slider_idx
    if event.key in ['up', 'down']:
        active_slider_idx = (active_slider_idx + (1 if event.key=='down' else -1)) % len(sliders)
        update_slider_visuals()
    elif event.key in ['left', 'right']:
        s = sliders[active_slider_idx]
        step = (s.valmax - s.valmin) * 0.02
        new_val = s.val + (step if event.key=='right' else -step)
        s.set_val(max(s.valmin, min(s.valmax, new_val)))

fig.canvas.mpl_connect('key_press_event', on_key_press)
update_slider_visuals()


# --- SIMULATION LOOP ---
max_points = 20000 
dt = 0.5
history_V = np.full(max_points, -60.0)
history_w = np.full(max_points, 0.01)
history_z = np.full(max_points, 0.0)
history_noise = np.full(max_points, 0.0)
time_axis = np.arange(max_points) * dt

state = [-60.0, 0.01, 0.1] 

# Burst Stats
burst_intervals = []
last_burst_time = 0
in_burst = False
sim_time = 0

def update(frame):
    global state, history_V, history_z, history_w, history_noise, sim_time, last_burst_time, in_burst, burst_intervals
    # 1. Physics Step (Multiple sub-steps for stability)
    steps = 40
    f_V, f_w, f_z, f_noise = np.zeros(steps), np.zeros(steps), np.zeros(steps), np.zeros(steps)
    I_base, eps, noise = s_I.val, s_Eps.val, s_Noise.val
    for i in range(steps):
        state = sde_step(state, dt, I_base, eps, noise)
        f_V[i], f_w[i], f_z[i] = state
        f_noise[i] = np.random.normal(0, 1) * noise * 10 # Visual noise scaling
        
        sim_time += dt

        # Burst Detection (Simple Threshold)
        if not in_burst and state[0] > -20:
            in_burst = True
            if last_burst_time > 0:
                interval = sim_time - last_burst_time
                if interval > 200: # Filter glitches
                    burst_intervals.append(interval)
                    if len(burst_intervals) > 100: burst_intervals.pop(0)
            last_burst_time = sim_time
        elif in_burst and state[0] < -40:
            in_burst = False

    # 2. Update History Arrays
    history_V = np.roll(history_V, -steps)
    history_w = np.roll(history_w, -steps)
    history_z = np.roll(history_z, -steps)
    history_noise = np.roll(history_noise, -steps)
    
    history_V[-steps:] = f_V
    history_w[-steps:] = f_w
    history_z[-steps:] = f_z
    history_noise[-steps:] = f_noise

    # 3. Update Visu
    # Update Status Text Box based on state
    if in_burst:
        text_status_box.set_text("ACTIVE (Bursting)")
        text_status_box.set_color('red')
        text_status_box.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='red'))
    else:
        text_status_box.set_text("SILENT (Recovery)")
        text_status_box.set_color('blue')
        text_status_box.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='blue'))

    # A. Left Column (Main Views)
    phase_len = 3000
    
    if view_mode == '2D_Slow':
        # Update Trajectory
        line_traj_2d.set_data(history_z[-phase_len:], history_V[-phase_len:])
        point_curr_2d.set_data([state[2]], [state[0]])
        
        # Recalculate S-curve for current I_base
        Vs_c, Zs_c = get_z_nullcline_v(I_base)
        valid = (Zs_c > -0.2) & (Zs_c < 1.2)
        
        if np.any(valid):
            valid_Vs = Vs_c[valid]
            valid_Zs = Zs_c[valid]
            line_equil.set_data(valid_Zs, valid_Vs)

            # Find Bifurcation Knees (Min/Max Z)
            idx_sn = np.argmin(valid_Zs)
            idx_hopf = np.argmax(valid_Zs)
            
            v_sn_val = valid_Vs[idx_sn]
            z_sn_val = valid_Zs[idx_sn]
            v_hopf_val = valid_Vs[idx_hopf]
            z_hopf_val = valid_Zs[idx_hopf]
            
            # Update Markers
            point_sn.set_data([z_sn_val], [v_sn_val])
            point_hopf.set_data([z_hopf_val], [v_hopf_val])
            
            # Update Threshold Lines
            line_thresh_sn.set_data([-0.1, 1.1], [v_sn_val, v_sn_val])
            line_thresh_hopf.set_data([-0.1, 1.1], [v_hopf_val, v_hopf_val])
            
            # Update labels near lines
            text_sn_label.set_position((0.02, v_sn_val + 2))
            text_hopf_label.set_position((0.02, v_hopf_val + 2))
            
    elif view_mode == '3D':
        line_traj_3d.set_data(history_z[-phase_len:], history_w[-phase_len:])
        line_traj_3d.set_3d_properties(history_V[-phase_len:])
        point_curr_3d.set_data([state[2]], [state[1]])
        point_curr_3d.set_3d_properties([state[0]])
        ax_phase_3d.view_init(elev=20, azim=(frame * 0.5) % 360)
        
    elif view_mode == '2D_Fast':
        line_traj_fast.set_data(history_V[-1000:], history_w[-1000:])
        point_curr_fast.set_data([state[0]], [state[1]])
        # Dynamic Nullcline (depends on current z)
        w_null = get_v_nullcline_w(v_range_fast, state[2], I_base)
        line_v_null_fast.set_data(v_range_fast, w_null)

    # B. Right Column Analysis (Downsampled)
    ds = 20
    sl = slice(0, max_points, ds)
    
    line_volt.set_data(time_axis[sl], history_V[sl])
    line_slow.set_data(time_axis[sl], history_z[sl])
    line_noise.set_data(time_axis[sl], history_noise[sl])

    # C. Histogram & CV
    if len(burst_intervals) > 2:
        # Full redraw of hist is expensive, done only when necessary or every N frames
        if frame % 5 == 0: 
            ax_ibi.cla()
            ax_ibi.set_title("IBI Distribution (Regularity)", fontsize=10, fontweight='bold')
            ax_ibi.set_ylabel("Count")
            ax_ibi.set_xlabel("Interval (ms)")
            ax_ibi.grid(True, alpha=0.3)
            
            ax_ibi.hist(burst_intervals, bins=15, color='steelblue', alpha=0.7, edgecolor='white')
            
            mean = np.mean(burst_intervals)
            std = np.std(burst_intervals)
            cv = std / mean if mean > 0 else 0
            
            ax_ibi.text(0.95, 0.85, f"CV: {cv:.3f}", 
                        transform=ax_ibi.transAxes, ha='right', va='center',
                        fontsize=10, fontweight='bold', color='green' if cv < 0.1 else 'orange',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    return line_volt,

def reset(event):
    global state, burst_intervals, last_burst_time, sim_time
    state = [-60.0, 0.01, 0.1]
    burst_intervals = []
    last_burst_time = sim_time

btn_reset.on_clicked(reset)

ani = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
plt.show()
