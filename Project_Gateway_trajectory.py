# By: Maans Rasmussen
from Orbital_functions_v4 import *
from mpl_toolkits.basemap import Basemap
from datetime import datetime as dt
import datetime
import folium
## Constants
mu_earth = 398600.4415      # [km^3/s^2]
R_earth = 6378.1363         # [km]
e_earth = 0.081819221456    # Eccentricity of Earth
R_sun = 696340              # [km]
R_moon = 1737.1             # [km]
J2 = 0.0002027              # for the moon
C_D = 2.0                   # drag coefficient
C_R = 1.5                   # Reflectivity  coefficient
mass_sat = 39000            # kg
A_cross_sat = 25            # m^2
A_div_m = A_cross_sat \
          / mass_sat  # [m^2/kg^]
mu_sun = 1.32 * 10 ** 11    # [km^3/s^2]
mu_moon = 4903              # [km^3/s^2]
delta_AT = 37               # [sec]

D2R_ = np.pi/180.0          # conversion from deg to rad


### Initial Orbit Parameters
# Circular inclined orbit
#special_case = "None"
#t_p = 0             # 2020-01-02, 00:00:00
#a = 36600        # [km]  - Semi-Major Axis
#e = 0.908
#i = 90               # [deg] - Inclination
#RAAN = 0            # [deg] - Right Ascension of the Ascending Node
#w = 90               # [deg] - Argument of periapsis
#nu = 0              # [deg] - true anomaly
#COE = np.array([a, e, i * D2R_, RAAN * D2R_, w * D2R_, nu])

# Circular polar orbit
special_case = "circular_inclined"
t_p = 0             # 2020-01-02, 00:00:00
a = 1838    # km
e = 0.0056
i = 90
i_i = 90 * D2R_
RAAN = -34.5
w = 45
nu = 0              # [deg] - true anomaly
COE = np.array([a, e, i * D2R_, RAAN * D2R_, w * D2R_, nu])

## Calculations
print("\nOrbit information: ")
n_T = np.sqrt(mu_moon / (a ** 3))       # mean motion
P_T = 2 * np.pi / n_T                    # period [s]
print("Period time: ", P_T, " sec\n", P_T/(60 * 60 * 24), " Days")

## Time
dUT1 = 0
dAT = 37    # sec
# Start of Epoch
UTC = [0, 0, 0]
date = [2020, 1, 2]
datetime_start = datetime.datetime(year=date[0], month=date[1], day=date[2], hour=UTC[0], minute=UTC[1], second=UTC[2], microsecond=0)
JD_UTC = UTCtoJD(date=date, UTC=UTC)          # initial JD (start of Epoch)
JD_UT1 = JD_UTC + dUT1 / 86400
JD_TAI = JD_UT1 + dAT / 86400
JD_TT = JD_TAI + 32.184 / 86400
T_TT = (JD_TT - 2451545) / 36525

######################################################################################################################
final_time_sec = 86400 * 31 #5 * P_T  # P_T * 5  # (end_year - start_year) * 365.2422 * 86400       # P_T * 5    # final time [sec]
time_step_sec = 10     # [sec] - half a day
T_sec = np.arange(0, final_time_sec + time_step_sec, time_step_sec)
###################################################################################################################
plot_3D_animation = True
plot_2D_animation = False
plot_ground_track = False
plot_2D = False
save_gif = False
save_every_n_data = 25   # 60 * 60 * 24   # 60 for 2D plots
lim = a * 1.1  # 75000
scale = 8
Title_3D = "Gateway in circular polar orbit"
Title_2D = "Gateway in circular polar orbit"
save_2D_ani = 'Gateway_orbit_2D.gif'
Title_pertb = "Magnitudes of the difference in position for different perturbations"
###################################################################################################################

#RV = COEstoRV(COE, mu_moon)    # converts from COE to r,v [km, km/s]
RV_all = propogate_orbit_ode_Moon_orbit(COE, T_sec, mu_moon, R_moon, R_sun, mu_earth, mu_sun, A_div_m,
                                                C_R, J2, JD_UT1, delta_AT, PERTB="all", special_case=special_case)
RV_2_body = propogate_orbit_ode_Moon_orbit(COE, T_sec, mu_moon, R_moon, R_sun, mu_earth, mu_sun, A_div_m,
                                                C_R, J2, JD_UT1, delta_AT, PERTB="2_body", special_case=special_case)
RV_SRP = propogate_orbit_ode_Moon_orbit(COE, T_sec, mu_moon, R_moon, R_sun, mu_earth, mu_sun, A_div_m,
                                                C_R, J2, JD_UT1, delta_AT, PERTB="SRP", special_case=special_case)
RV_J2 = propogate_orbit_ode_Moon_orbit(COE, T_sec, mu_moon, R_moon, R_sun, mu_earth, mu_sun, A_div_m,
                                                C_R, J2, JD_UT1, delta_AT, PERTB="J2", special_case=special_case)
RV_3rd = propogate_orbit_ode_Moon_orbit(COE, T_sec, mu_moon, R_moon, R_sun, mu_earth, mu_sun, A_div_m,
                                                C_R, J2, JD_UT1, delta_AT, PERTB="3rd_body", special_case=special_case)

v_i = np.linalg.norm(RV_all[0, 3:])
print("v_i: ", v_i)
# Creates arrays
R_ani_x = []
R_ani_y = []
R_ani_z = []
T_arr = []
R_EARTH_x = []
R_EARTH_y = []
R_EARTH_z = []
COE_arr = []

SRP_MAG = []
Two_body_MAG = []
J2_MAG = []
third_body_MAG = []

oneday = 24 * 60 * 60
frames_len = len(RV_all[:, 0:3])
ii = 0
# form: [[year, month, day, MJD], xp, yp, dUT1, LOD, dPsi, dEpsilon, dX, DY, dAT]
# form: [['1991', '07', '18', '48455'], '0.035965', '0.572780', '0.2029027', '0.0015752 -0.013801 -0.004509 -0.000092 -0.000072', '26\n']
for i in range(frames_len):
    # Goes though all vectors, transforms to lat, lon and saves them
    r_GCRF = RV_all[i, 0:3]
    v_GCRF = RV_all[i, 3:]
    #print("v_i: ", np.linalg.norm(RV_all[0, 3:]))
    # get earth position
    JD_TDB = JD_UTC2JD_TDB(JD_UT1, delta_AT)
    JD_UT1 += time_step_sec / 86400
    r_earth_moon = getMoonPos_wrt_Earth(JD_TDB)
    r_moon_earth = - r_earth_moon
    if i % save_every_n_data == 0:
        # saves every 10th min (600 sec)
        R_ani_x.append(r_GCRF[0])   # [km]
        R_ani_y.append(r_GCRF[1])   # [km]
        R_ani_z.append(r_GCRF[2])   # [km]
        R_EARTH_x.append(r_moon_earth[0]/scale)
        R_EARTH_y.append(r_moon_earth[1]/scale)
        R_EARTH_z.append(r_moon_earth[2]/scale)

        J2_MAG.append(1000 * np.linalg.norm(RV_J2[i, 0:3] - (RV_all[i, 0:3])))
        SRP_MAG.append(1000 * np.linalg.norm(RV_SRP[i, 0:3] - (RV_all[i, 0:3])))
        third_body_MAG.append(1000 * np.linalg.norm(RV_3rd[i, 0:3] - (RV_all[i, 0:3])))
        Two_body_MAG.append(1000 * np.linalg.norm(RV_2_body[i, 0:3] - (RV_all[i, 0:3])))

        T_arr.append(ii)
        COEs = RVtoCOE_special(r_GCRF, v_GCRF, mu_moon, special_case)
        COE_arr.append(COEs)
        ii += 1

T_arr = np.array(T_arr)
COE_arr = np.array(COE_arr)

# Calculates dv required to maintain initial orbit
# in rad!
i_f = COE_arr[-1, 2]
dRAAN = (360 + RAAN) * D2R_ - COE_arr[-1, 3]
theta = np.arccos(np.cos(i_i)*np.cos(i_f) + np.sin(i_i)*np.sin(i_f)*np.cos(dRAAN))
delta_v = 2*v_i*np.sin(theta/2)
print("delta_v: ", delta_v)
print("theta: ", theta / D2R_)


if plot_3D_animation:
    # Set 3D axes limits
    #x_max = int(max(R_EARTH_x))
    #y_max = int(max(R_EARTH_y))
    #z_max = int(max(R_EARTH_z))
    #lim = int(max(x_max, y_max, z_max))
    # Plots the animation
    R_animate = [R_ani_x, R_ani_y, R_ani_z, R_EARTH_x, R_EARTH_y, R_EARTH_z, T_arr]
    R_animate = np.array(R_animate)
    #print("R_animate: ", R_animate)
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.plot(R_EARTH[:,0],R_EARTH[:,1], R_EARTH[:,2], '+', color="green")

    moon, = ax.plot([], [], [], '8', color="grey", label="Moon")
    orbit, = ax.plot([], [], [])
    satellite, = ax.plot([], [], [], 'o', color='red', label="Gateway")
    earth, = ax.plot([], [], [], 'o', color='green', markersize=1)#, label="Earth")
    time_text = ax.text(0.02, 0.95, 0, '', transform=ax.transAxes)

    def init_fig():
        """Initialize the figure, used to draw the first
        frame for the animation. Sets the background.
        """
        # Set the axis and plot titles
        orbit, = ax.plot([], [], [])
        satellite, = ax.plot([], [], [], 'o', color='red')
        earth, = ax.plot([], [], [], 'o', color='green')
        time_text.set_text('')
        ax.set_title(Title_3D, fontsize=22)
        ax.set_xlim3d([-lim, lim])
        ax.set_xlabel('I\n[km]')
        ax.set_ylim3d([-lim, lim])
        ax.set_ylabel('J\n[km]')
        ax.set_zlim3d([-lim, lim])
        ax.set_zlabel('K\n[km]')
        # plot Earth

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = R_moon * np.outer(np.cos(u), np.sin(v))
        y = R_moon * np.outer(np.sin(u), np.sin(v))
        z = R_moon * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color="grey", label="Moon", linewidth=0.3, rstride=7, cstride=7)
        # Must return the list of artists, but we use a pass
        # through so that they aren't created multiple times
        return orbit, satellite, earth, time_text

    def animate(i, R_array, orbit, satellite, earth, time_text):
        orbit.set_data(R_array[0][:i], R_array[1][:i])
        orbit.set_3d_properties(R_array[2][:i])
        satellite.set_data(R_array[0][i], R_array[1][i])
        satellite.set_3d_properties(R_array[2][i])
        time_text.set_text("Days: %.1f" % (R_array[6][i] * save_every_n_data * time_step_sec/86400))

        #earth.set_data(R_array[3][:i], R_array[4][:i])
        #earth.set_3d_properties(R_array[5][:i])

        return orbit, satellite, earth, time_text

    ani = animation.FuncAnimation(fig, animate, init_func=init_fig, interval=1, frames=frames_len-1, repeat=True,
                                       fargs=(R_animate, orbit, satellite, earth, time_text), blit=True)
    plt.legend()
    if save_gif:
        ani.save('Gateway_orbit.gif', writer='pillow', fps=30)

    plt.show()

if plot_2D_animation:

    # Plots the animation
    R_animate = [R_ani_x, R_ani_y, R_ani_z, T_arr]
    R_animate = np.array(R_animate)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

    #sat, = ax.plot([], [], label="Satellite")
    trail1, = ax1.plot([], [], '.', color='red', markersize=1, label="Gateway")
    trail2, = ax2.plot([], [], '.', color='red', markersize=1, label="Gateway")
    trail3, = ax3.plot([], [], '.', color='red', markersize=1, label="Gateway")
    time_text = ax1.text(0.05, 0.1, '', transform=ax1.transAxes)

    def init():
        trail1.set_data([], [])
        trail2.set_data([], [])
        trail3.set_data([], [])
        font = {'size': 12}
        plt.rc('font', **font)
        time_text.set_text('')
        ax1.legend()
        ax1.set_xlim([-lim, lim])
        ax1.set_xlabel('I [km]')
        ax1.set_ylim([-lim, lim])
        ax1.set_ylabel('J [km]')

        ax2.set_title(Title_2D, fontsize=18, weight="bold")
        ax2.set_xlim([-lim, lim])
        ax2.set_xlabel('I [km]')
        ax2.set_ylim([-lim, lim])
        ax2.set_ylabel('K [km]')

        ax3.set_xlim([-lim, lim])
        ax3.set_xlabel('J [km]')
        ax3.set_ylim([-lim, lim])
        ax3.set_ylabel('K [km]')

        ax1.set(adjustable='box-forced', aspect='equal')
        ax2.set(adjustable='box-forced', aspect='equal')
        ax3.set(adjustable='box-forced', aspect='equal')
        return trail1, trail2, trail3

    # animation function.  This is called sequentially
    def animate(i, R_animate, trail1, trail2, trail3, time_text):
        i = i-1
        x = R_animate[0][0:i]
        y = R_animate[1][0:i]
        z = R_animate[2][0:i]
        trail1.set_data(x, y)
        trail2.set_data(x, z)
        trail3.set_data(y, z)
        time_text.set_text("Days propagated: %.1f" % (R_animate[3][i] * save_every_n_data * time_step_sec / 86400))
        return trail1, trail2, trail3, time_text


    # call the animator.  blit=True means only re-draw the parts that have changed.
    ani = animation.FuncAnimation(fig, animate, init_func=init, fargs=(R_animate, trail1, trail2, trail3, time_text),
                                        interval=10, blit=True, repeat=False, frames=frames_len-1)


    if save_gif:
        ani.save(save_2D_ani, writer='imagemagick', fps=30)
    plt.show()

if plot_2D:

    # Plot the largest magnitudes of the differences
    DIFF_MAG = {'two-body': Two_body_MAG[-1], 'third-body': third_body_MAG[-1], 'SRP': SRP_MAG[-1],
                'J2': J2_MAG[-1]}
    sort = sorted(DIFF_MAG.items(), key=lambda x: x[1])
    print("Sorted Diff_Mag (last elements): ", sort)

    # Plots the animation
    R_animate = [R_ani_x, R_ani_y, R_ani_z]
    R_animate = np.array(R_animate)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

    trail1, = ax1.plot([], [], '.', color='red', markersize=1, label="Gateway")
    trail2, = ax2.plot([], [], '.', color='red', markersize=1, label="Gateway")
    trail3, = ax3.plot([], [], '.', color='red', markersize=1, label="Gateway")
    time_text = fig.text(0.05, 0.1, "Days propagated: %.1f" % (T_arr[-1] * save_every_n_data * time_step_sec / 86400))


    x = R_animate[0][0:i]
    y = R_animate[1][0:i]
    z = R_animate[2][0:i]
    trail1.set_data(x, y)
    trail2.set_data(x, z)
    trail3.set_data(y, z)

    font = {'size': 12}
    plt.rc('font', **font)
    ax1.legend()
    ax1.set_xlim([-lim, lim])
    ax1.set_xlabel('I')
    ax1.set_ylim([-lim, lim])
    ax1.set_ylabel('J')

    ax2.set_title(Title_2D, fontsize=18, weight="bold")
    ax2.set_xlim([-lim, lim])
    ax2.set_xlabel('I')
    ax2.set_ylim([-lim, lim])
    ax2.set_ylabel('K')

    ax3.set_xlim([-lim, lim])
    ax3.set_xlabel('J')
    ax3.set_ylim([-lim, lim])
    ax3.set_ylabel('K')

    ax1.set(adjustable='box-forced', aspect='equal')
    ax2.set(adjustable='box-forced', aspect='equal')
    ax3.set(adjustable='box-forced', aspect='equal')

    #plt.show()
    T_arr = T_arr * save_every_n_data * time_step_sec / 86400
    plt.figure()
    plt.plot(T_arr, COE_arr[:, 0])
    plt.xlabel('Days since Start of Epoch')
    plt.ylabel('Semimajor axis [km]')
    plt.title('Semimajor axis vs. Time')
    plt.figure()
    plt.plot(T_arr, COE_arr[:, 1])
    plt.xlabel('Days since Start of Epoch')
    plt.ylabel('Eccentricity')
    plt.title('Eccentricity vs. Time')
    plt.figure()
    plt.plot(T_arr, (COE_arr[:, 2] * 180 / np.pi))
    plt.xlabel('Days since Start of Epoch')
    plt.ylabel('Inclination [degrees]')
    plt.title('Inclination vs. Time')
    plt.figure()
    plt.plot(T_arr, (COE_arr[:, 3] * 180 / np.pi))
    plt.xlabel('Days since Start of Epoch')
    plt.ylabel('RAAN [degrees]')
    plt.title('RAAN vs. Time')
    plt.figure()
    plt.plot(T_arr, (COE_arr[:, 4] * 180 / np.pi))
    plt.xlabel('Days since Start of Epoch')
    plt.ylabel('Argument of latitude [degrees]')
    plt.title('Argument of latitude vs. Time')

    plt.figure()

    plt.plot(T_arr, Two_body_MAG, label="Two-body", linewidth=3)
    plt.plot(T_arr, J2_MAG, label="Two-body + J2")
    plt.plot(T_arr, SRP_MAG, label="Two-body + SRP")
    plt.plot(T_arr, third_body_MAG, label="Two-body + Sun + Earth")
    font = {'size': 12}
    plt.rc('font', **font)
    plt.ylim(bottom=1)
    plt.yscale("log")
    plt.title(Title_pertb, fontsize=18, weight="bold")
    plt.ylabel("Positional Difference [m]", fontsize=15)
    plt.xlabel("Time from Epoch start [days]", fontsize=15)
    plt.legend(loc=4)
    plt.grid(True)
    plt.show()

    plt.show()