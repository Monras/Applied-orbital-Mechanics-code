#By Maans Rasmussen
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math as m
from decimal import *
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
## Constants
D2R_ = np.pi / 180.0
R2D_ = 180.0 / np.pi
Arcs2Rad_ = np.pi / 648000

def RVtoCOEs(r_vec, v_vec, mu):
    """Converts cartesian coordinates to orbital elements,
     returns one array of COEs"""
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    # Angular momentum vector
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    k_vec = np.array([0, 0, 1])
    # Line of nodes vector
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)
    # Eccentricity vector
    e_vec = (1 / mu) * ((v ** 2 - mu / r) * r_vec - ((np.dot(r_vec, v_vec)) * v_vec))
    e = np.linalg.norm(e_vec)
    if abs(e) < 10 ** -12:
        # circular orbit!
        e = 0
    # Calculates the energy
    epsilon = 0.5*v**2 - mu/r

    # calculates the semi-major axis
    if epsilon < 0:
        a = - mu / (2 * epsilon)
    elif epsilon == 0:
        # circular orbit
        a = np.infty
        p = a*(1 - e**2)
    else:
        a = mu / (2 * epsilon)

    # Calculates true anomaly (v)
    if e == 0:
        # circular orbit (inclined or not), true anomaly not defined!
        print("true anomaly not defient, e = 0")
        arg = np.inf
    else:
        arg = (np.dot(r_vec, e_vec))/(r*e)

    # simplifies/removes computational errors
    if np.abs(arg - 1) < 10**-12 and arg > 1:
        arg = 1
    elif np.abs(arg + 1) < 10**-12 and arg < -1:
        arg = -1
    theta = np.arccos(arg)

    if np.dot(r_vec, e_vec) < 0:
        # checks ambiguity
        theta = 2*np.pi - theta

    # Calculates the inclination (i)
    arg = (np.dot(h_vec, k_vec) / h)
    # simplifies/removes computational errors
    if np.abs(arg - 1) < 10**-12 and arg > 1:
        arg = 1
    elif np.abs(arg + 1) < 10**-12 and arg < -1:
        arg = -1
    i = np.arccos(arg)



    if i == 0:
        raise ValueError("non-circular (elliptical) equatorial orbit! Line of nodes and RAAN is not defined!")
        # omega_true
    elif i == 0 and e == 0:
        raise ValueError("Circular equatorial orbit! Line of nodes, RAAN, true anomaly & argument of periapsis"
                         " is not defined!")
        # lambda_true

    # Calculate RAAN
    arg = n_vec[0]/n
    # simplifies/removes computational errors
    if np.abs(arg - 1) < 10**-12 and arg > 1:
        arg = 1
    elif np.abs(arg + 1) < 10**-12 and arg < -1:
        arg = -1
    RAAN = np.arccos(arg)
    # checks for ambiguity
    if n_vec[1] < 0:
        RAAN = 2*np.pi - RAAN

    # Calculate argument of periapsis, omega
    if e == 0:
        # circular orbit (inclined or not), true anomaly not defined!
        raise ValueError("Circular orbit! argument of periapsis is not defined!")
    else:
        arg = np.dot(n_vec, e_vec)/(n*e)

    if np.abs(arg - 1) < 10**-12 and arg > 1:
        arg = 1
    elif np.abs(arg + 1) < 10**-12 and arg < -1:
        arg = -1
    omega = np.arccos(arg)
    if e_vec[2] < 0:
        omega = 2*np.pi - omega

    # True longitude of periapsis, lambda true
    arg = r_vec[0]/r
    if np.abs(arg - 1) < 10**-12 and arg > 1:
        arg = 1
    elif np.abs(arg + 1) < 10**-12 and arg < -1:
        arg = -1
    l = np.arccos(arg)
    if r_vec[1] < 0:
        l = 2*np.pi - l

    # Argument of latitude, u
    arg = np.dot(n_vec, r_vec)/r
    if np.abs(arg - 1) < 10**-12 and arg > 1:
        arg = 1
    elif np.abs(arg + 1) < 10**-12 and arg < -1:
        arg = -1
    u = np.arccos(arg)
    if r_vec[2] < 0:
        u = 2*np.pi - u
    if e > 0:
        u = theta + omega

    # Longitude of periapsis
    if e == 0:
        arg = np.infty
    else:
        arg = e_vec[0]/e

    if np.abs(arg - 1) < 10**-12 and arg > 1:
        arg = 1
    elif np.abs(arg + 1) < 10**-12 and arg < -1:
        arg = -1
    omega_bar = np.arccos(arg)
    if e_vec[1] < 0:
        omega_bar = 2*np.pi - omega_bar

    # Singularities check
    if e < 10**-12 and i < 10**-12:
        omega = 0
        RAAN = 0
        theta = l
    elif e < 10**-12:
        omega = 0
        theta = u
    elif i < 10**-12:
        RAAN = 0
        omega = omega_bar

    COEs = np.array([a, e, i, RAAN, omega, theta])  # output in rad
    return COEs

def RVtoCOE_special(r_vec, v_vec, mu, special_case):
    """Converts cartesian coordinates to orbital elements,
     returns one array of COEs"""

    # special_case_arr = [None, "Elliptical equatorial", Circular inclined, Circular equatorial]
    if special_case == "None":
        ## No special case
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        # Angular momentum vector
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        k_vec = np.array([0, 0, 1])
        # Line of nodes vector
        n_vec = np.cross(k_vec, h_vec)
        n = np.linalg.norm(n_vec)
        # Eccentricity vector
        e_vec = (1 / mu) * ((v ** 2 - mu / r) * r_vec - ((np.dot(r_vec, v_vec)) * v_vec))
        e = np.linalg.norm(e_vec)
        if abs(e) < 10 ** -12:
            # circular orbit!
            e = 0
        # Calculates the energy
        epsilon = 0.5 * v ** 2 - mu / r

        # calculates the semi-major axis
        if epsilon < 0:
            a = - mu / (2 * epsilon)
        elif epsilon == 0:
            # circular orbit
            a = np.infty
            p = a * (1 - e ** 2)
        else:
            a = mu / (2 * epsilon)

        # Calculates true anomaly (v)
        if e != 0:
            arg = (np.dot(r_vec, e_vec)) / (r * e)
        else:
            raise ValueError("eccentricity is zero! in the wrong special case!")

        # simplifies/removes computational errors
        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        theta = np.arccos(arg)
        if np.dot(r_vec, e_vec) < 0:
            # checks ambiguity
            theta = 2 * np.pi - theta

        # Calculates the inclination (i)
        arg = (np.dot(h_vec, k_vec) / h)
        # simplifies/removes computational errors
        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        i = np.arccos(arg)

        if i == 0:
            raise ValueError("non-circular (elliptical) equatorial orbit! Line of nodes and RAAN is not defined!")
            # omega_true
        elif i == 0 and e == 0:
            raise ValueError("Circular equatorial orbit! Line of nodes, RAAN, true anomaly & argument of periapsis"
                             " is not defined!")
            # lambda_true

        # Calculate RAAN
        arg = n_vec[0] / n
        # simplifies/removes computational errors
        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        RAAN = np.arccos(arg)
        # checks for ambiguity
        if n_vec[1] < 0:
            RAAN = 2 * np.pi - RAAN

        # Calculate argument of periapsis, omega
        if e == 0:
            # circular orbit (inclined or not), true anomaly not defined!
            raise ValueError("Circular orbit! argument of periapsis is not defined!")
        else:
            arg = np.dot(n_vec, e_vec) / (n * e)

        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        omega = np.arccos(arg)
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
        COEs = np.array([a, e, i, RAAN, omega, theta])  # output in rad
    elif special_case == "elliptical_equatorial":
        print(special_case)
    elif special_case == "circular_inclined":
        #print(special_case)
        ## e = 0 and i != 0
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        # Angular momentum vector
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        k_vec = np.array([0, 0, 1])

        # Line of nodes vector
        n_vec = np.cross(k_vec, h_vec)
        n = np.linalg.norm(n_vec)

        # Eccentricity vector
        e_vec = (1 / mu) * ((v ** 2 - mu / r) * r_vec - ((np.dot(r_vec, v_vec)) * v_vec))
        e = np.linalg.norm(e_vec)
        if abs(e) < 10 ** -12:
            # circular orbit!
            e = 0
        # Calculates the energy
        epsilon = 0.5 * v ** 2 - mu / r

        # calculates the semi-major axis
        if epsilon < 0:
            a = - mu / (2 * epsilon)
        elif epsilon == 0:
            # circular orbit
            a = np.infty
            p = a * (1 - e ** 2)
        else:
            a = mu / (2 * epsilon)

        # Calculates the inclination (i)
        arg = (np.dot(h_vec, k_vec) / h)
        # simplifies/removes computational errors
        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        i = np.arccos(arg)

        # Calculate RAAN
        arg = n_vec[0] / n
        # simplifies/removes computational errors
        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        RAAN = np.arccos(arg)
        # checks for ambiguity
        if n_vec[1] < 0:
            RAAN = 2 * np.pi - RAAN

        # Argument of latitude, u
        arg = np.dot(n_vec, r_vec) / (r * np.linalg.norm(n_vec))
        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        u = np.arccos(arg)
        if r_vec[2] < 0:
            u = 2 * np.pi - u
        COEs = np.array([a, e, i, RAAN, u])  # output in rad
    elif special_case == "circular_equatorial":
        print(special_case)
        ## No special case
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        # Angular momentum vector
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        k_vec = np.array([0, 0, 1])
        # Line of nodes vector
        n_vec = np.cross(k_vec, h_vec)
        n = np.linalg.norm(n_vec)
        # Eccentricity vector
        e_vec = (1 / mu) * ((v ** 2 - mu / r) * r_vec - ((np.dot(r_vec, v_vec)) * v_vec))
        e = np.linalg.norm(e_vec)
        if abs(e) < 10 ** -12:
            # circular orbit!
            e = 0
        # Calculates the energy
        epsilon = 0.5 * v ** 2 - mu / r

        # calculates the semi-major axis
        if epsilon < 0:
            a = - mu / (2 * epsilon)
        elif epsilon == 0:
            # circular orbit
            a = np.infty
            p = a * (1 - e ** 2)
        else:
            a = mu / (2 * epsilon)

        # Calculates true anomaly (v)
        if e != 0:
            arg = (np.dot(r_vec, e_vec)) / (r * e)
        else:
            raise ValueError("eccentricity is zero! in the wrong special case!")

        # simplifies/removes computational errors
        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        theta = np.arccos(arg)
        if np.dot(r_vec, e_vec) < 0:
            # checks ambiguity
            theta = 2 * np.pi - theta

        # Calculates the inclination (i)
        arg = (np.dot(h_vec, k_vec) / h)
        # simplifies/removes computational errors
        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        i = np.arccos(arg)

        if i == 0:
            raise ValueError("non-circular (elliptical) equatorial orbit! Line of nodes and RAAN is not defined!")
            # omega_true
        elif i == 0 and e == 0:
            raise ValueError("Circular equatorial orbit! Line of nodes, RAAN, true anomaly & argument of periapsis"
                             " is not defined!")
            # lambda_true

        # Calculate RAAN
        arg = n_vec[0] / n
        # simplifies/removes computational errors
        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        RAAN = np.arccos(arg)
        # checks for ambiguity
        if n_vec[1] < 0:
            RAAN = 2 * np.pi - RAAN

        # Calculate argument of periapsis, omega
        if e == 0:
            # circular orbit (inclined or not), true anomaly not defined!
            raise ValueError("Circular orbit! argument of periapsis is not defined!")
        else:
            arg = np.dot(n_vec, e_vec) / (n * e)

        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        omega = np.arccos(arg)
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega

        # True longitude of periapsis, lambda true
        arg = r_vec[0] / r
        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        l = np.arccos(arg)
        if r_vec[1] < 0:
            l = 2 * np.pi - l

        # Argument of latitude, u
        arg = np.dot(n_vec, r_vec) / r
        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        u = np.arccos(arg)
        if r_vec[2] < 0:
            u = 2 * np.pi - u
        if e > 0:
            u = theta + omega

        # Longitude of periapsis
        if e == 0:
            arg = np.infty
        else:
            arg = e_vec[0] / e

        if np.abs(arg - 1) < 10 ** -12 and arg > 1:
            arg = 1
        elif np.abs(arg + 1) < 10 ** -12 and arg < -1:
            arg = -1
        omega_bar = np.arccos(arg)
        if e_vec[1] < 0:
            omega_bar = 2 * np.pi - omega_bar

        # Singularities check
        if e < 10 ** -12 and i < 10 ** -12:
            omega = 0
            RAAN = 0
            theta = l
        elif e < 10 ** -12:
            omega = 0
            theta = u
        elif i < 10 ** -12:
            RAAN = 0
            omega = omega_bar

    return COEs

def COEstoRV(COEs, mu):
    """Converts orbital elements to cartesian coordinates in IJK frame!,
    returns two 3x1 arrays"""

    # if circular equatorial
        # omega, RAAN = 0 & theta = lambda_true
    # if Circular inclined
        # omega = 0 and theta = u
    # if Elliptical inclined

    a = COEs[0]
    e = COEs[1]
    i = COEs[2]
    RAAN = COEs[3]
    omega = COEs[4]
    theta = COEs[5]
    # Semi-parameter, p
    p = a * (1 - e ** 2)
    # Coordinates in PQW frame
    temp1 = p*np.cos(theta)/(1 + e*np.cos(theta))
    temp2 = p*np.sin(theta)/(1 + e*np.cos(theta))
    r_vec_PQW = np.array([temp1, temp2, 0])
    v_vec_PQW = np.sqrt(mu/p)*np.array([-np.sin(theta), (e + np.cos(theta)), 0])

    RAAN = -RAAN
    omega = -omega
    i = -i

    R3_RAAN = np.array([[np.cos(RAAN), np.sin(RAAN), 0], [-np.sin(RAAN), np.cos(RAAN), 0], [0, 0, 1]])
    R3_omega = np.array([[np.cos(omega), np.sin(omega), 0], [-np.sin(omega), np.cos(omega), 0], [0, 0, 1]])
    R1_i = np.array([[1, 0, 0], [0, np.cos(i), np.sin(i)], [0, -np.sin(i), np.cos(i)]])

    # Create the transformation matrix from PQW to ijk
    Q_temp = np.dot(R3_RAAN, R1_i)
    Q = np.dot(Q_temp, R3_omega)

    # Singularitites
    """ 
    if e < 10**-12 and i < 10**-12:
        Q = np.eye(3)
    elif e < 10**-12:
        Q = np.matmul(R3_RAAN, R1_i)
    elif i < 10**-12:
        Q = R3_omega
    """
    r_IJK = np.dot(Q, r_vec_PQW)
    v_IJK = np.dot(Q, v_vec_PQW)

    # removes vary small numbers for the position and velocity (if smaller then nanometer)
    for ii in range(0, len(r_IJK)):
        if abs(r_IJK[ii]) < 10**-12:
            r_IJK[ii] = 0
    for ii in range(0, len(v_IJK)):
        if abs(v_IJK[ii]) < 10 ** -12:
            v_IJK[ii] = 0

    return r_IJK, v_IJK

def propogate_orbit_newt(COEs, T, mu):
    """propogates the orbit given COEs (orbit elements) and time"""
    tp = 0
    STOP = 1000000  # stops after 1 million iterations
    tol = np.exp(-10)

    a = COEs[0]
    e = COEs[1]
    i = COEs[2]
    RAAN = COEs[3]
    omega = COEs[4]
    theta = COEs[5]
    COEs_arr = []
    R_arr = []
    V_arr = []
    # Saves initial positions
    COEs = [a, e, i, RAAN, omega, theta]
    COEs_arr.append(COEs)
    r, v = COEstoRV(COEs, mu)
    R_arr.append(r)
    V_arr.append(v)

    n = np.sqrt(mu/(a**3))  # period
    M0 = n*(T[0] - tp)
    E = M0 + np.exp(-12)
    for t in T[1:]:
        counter = 0
        epsilon = 1
        Me = n*(t - tp)
        # solve for E with newtons method
        while epsilon > tol and counter < STOP:
            E_old = E
            E = Newtons(E, Me, e)
            epsilon = np.abs(E-E_old)
            counter += 1
        # Solving for the true anomaly
        theta = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
        COEs = [a, e, i, RAAN, omega, theta]
        COEs_arr.append(COEs)
        r, v = COEstoRV(COEs, mu)
        R_arr.append(r)
        V_arr.append(v)
    print("done!")
    print("M: ", Me)
    print("theta: ", theta*180/np.pi)
    return COEs_arr, R_arr, V_arr, T

def Newtons(E, Me, e):
    """Does one iteration on Newtons method"""
    f = E - e*np.sin(E) - Me
    f_prime = 1 - e*np.cos(E)
    E = E - f/f_prime
    return E

def propogate_orbit_ode(COEs, T, mu):
    """propogates the orbit given COEs and time vector"""
    tp = 0  # time of perihelion
    tol = 10**-10  # tolerance
    # unpacks the orbital elements
    a = COEs[0]
    e = COEs[1]
    i = COEs[2]
    RAAN = COEs[3]
    omega = COEs[4]
    theta = COEs[5]
    # arrays
    COEs_arr = []
    R_arr = []
    V_arr = []

    # Saves initial positions
    COEs = [a, e, i, RAAN, omega, theta]
    COEs_arr.append(COEs)
    r, v = COEstoRV(COEs, mu)
    R_arr.append(r)
    V_arr.append(v)
    r_transp = np.asarray(np.transpose(r))
    v_transp = np.asarray(np.transpose(v))
    Y0 = np.append(r_transp, v_transp)

    # Does the integrations
    Y_solv = integrate.odeint(function, Y0, T, args=(mu, ), rtol= 1e-10, atol= 1e-10)

    return Y_solv

def function(Y, t, mu):
    """Defines the ordinary differential equation/function of the two-body problem"""
    r = np.asarray(Y[0:3])
    r_mag = np.linalg.norm(r)
    v = np.asarray(Y[3:])
    Y = [v, - mu*r/(r_mag**3)]
    Y = np.append(Y[0], Y[1])
    return np.asarray(Y)

def UTCtoJD(date, UTC):
    """Converts UTC time to Modified Julian Date 2000"""
    Y = date[0]
    M = date[1]
    D = date[2]
    h = UTC[0]
    min = UTC[1]
    sec = UTC[2]
    JD_date = 367*Y - m.trunc((7/4)*(Y + m.trunc((M + 9)/12))) + m.trunc((275*M)/9) + D + 1721013.5
    JD_day = (1/24)*(h + (1/60)*(min + sec/60))
    JD = JD_date + JD_day

    return JD

def UTCtoMJD(date, UTC):
    """Converts UTC time to Modified Julian Date 2000"""
    Y = date[0]
    M = date[1]
    D = date[2]
    h = UTC[0]
    min = UTC[1]
    sec = UTC[2]
    JD_date = 367 * Y - m.trunc((7 / 4) * (Y + m.trunc((M + 9) / 12))) + m.trunc((275 * M) / 9) + D + 1721013.5
    JD_day = (1 / 24) * (h + (1 / 60) * (min + sec / 60))
    print("JD_day: ", JD_day)
    JD = JD_date + JD_day
    MJD = JD  # - 2400000.5
    return MJD

def DMStoRAD(D,M,S):
    """Transforms degree-arcminutes-acrseconds to radians"""
    RAD = (np.pi/180)*(D + M/60 + S/3600)     # (1/np.pi)*(D + M/60 + S/3600)
    return RAD

def RADtoDMS(RAD):
    """Transforms radians to degree-arcminutes-acrseconds"""
    temp = RAD*(180/np.pi)
    deg = m.trunc(temp)
    arcmin = m.trunc((temp - deg)*60)
    arcsec = (temp - deg - arcmin/60)*3600
    return [deg, arcmin, arcsec]

def HMStoRAD(H,M,S):
    """Transforms hour minutes seconds to radians"""
    rad = 15*(H + M/60 + S/3600)*np.pi/180
    return rad

def HMStoDEG(H,M,S):
    """Transforms hour minutes seconds to degrees"""
    rad = 15*(H + M/60 + S/3600)
    return rad

def JDtoGregorianDate(JD):
    """Transforms the date from JD to Gregorian Date (valid year 1900-2100)"""
    temp = 0
    sum_days = 0
    i = 0
    LMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # number of days each month
    T1900 = (JD - 2415019.5)/365.25
    Year = 1900 + m.trunc(T1900)
    LeapYrs = m.trunc((Year - 1900 -1)*0.25)
    Days = (JD - 2415019.5) - ((Year - 1900)*365 + LeapYrs)
    if Days < 1:
        Year = Year - 1
        LeapYrs = m.trunc((Year - 1900 - 1)*0.25)
        Days = (JD - 2415019.5) - ((Year - 1900)*365 + LeapYrs)
    if Year % 4 == 0:
        LMonth[2] = 29
    DayofYr = m.trunc(Days)
    for d in LMonth:
        temp += d
        i += 1
        if temp < DayofYr:
            sum_days += d
            continue
        else:
            break
    Mon = i
    Day = DayofYr - sum_days
    tao = (Days - DayofYr)*24
    h = m.trunc(tao)
    min = m.trunc((tao - h)*60)
    sec = (tao - h - min/60)*3600
    return [Year, Mon, Day], [h, min, sec]

def TTtoTAI(TT):
    """Terrestrial time JD2000 to atomic time"""
    temp = UTCtoJD([2000,1,1], [0,0,32.184])
    TAI = TT - (32.184/3600*24)
    return TAI

def UTCtoTAI(UTC):
    """Transforms UTC JD2000 to TAI"""
    TAI = UTC + 37/(3600*24)
    return TAI

def CONVTIME(UTC, delta_UT1, delta_AT):
    """Transforms yr, mo, day, UTC etc. to UT1, TAI, TT, T_UT1 """
    UT1 = UTC + delta_UT1
    TAI = UTC + delta_AT
    GPS = UTC + delta_AT - 19/(3600*24)
    TT = TAI + 32.184/(3600*24)

    return UT1, TAI, GPS, TT

def ITRFtoGCRF(r_ITRF, v_ITRF,  UTC, dUT1, x_p, y_p, theta_ERA, X, Y, dX, dY, s):
    """Transforms ITRF frame vectors to GCRF frame"""
    #UT1, TAI, GPS, TT = CONVTIME(UTC, dUT1, dAT)

    X = X + dX
    Y = Y + dY

    JDTT = UTCtoJD(UTC[0], UTC[1])
    T_TT = (JDTT - 2451545) / 36525


    a = 0.5 + 0.125*(X**2 + Y**2)
    LOD = dUT1
    omega_earth = [0, 0, 7.292115146706979*10**-5*(1 - LOD/86400)]


    # R matrix
    R = np.array([[np.cos(-theta_ERA), np.sin(-theta_ERA), 0], [-np.sin(-theta_ERA), np.cos(-theta_ERA), 0], [0, 0, 1]])  # R3 rotation

    s_prim = 0.000047*T_TT    # arcmin or rad?
    #s_prim = DMStoRAD(0, s_prim, 0)     # rad

    # W matrix
    R3_s_prim = np.array([[np.cos(-s_prim), np.sin(-s_prim), 0], [-np.sin(-s_prim), np.cos(-s_prim), 0], [0, 0, 1]])
    R1_yp = np.array([[1, 0, 0], [0, np.cos(y_p), np.sin(y_p)], [0, -np.sin(y_p), np.cos(y_p)]])
    R2_xp = np.array([[np.cos(x_p), 0, -np.sin(x_p)], [0, 1, 0], [np.sin(x_p), 0, np.cos(x_p)]])
    W_temp = np.dot(R3_s_prim, R2_xp)
    W = np.dot(W_temp, R1_yp)


    # PN matrix
    R3_s = np.array([[np.cos(s), np.sin(s), 0], [-np.sin(s), np.cos(s), 0], [0, 0, 1]])
    temp = np.array([[1 - a * X ** 2, -a * X * Y, X], [-a * X * Y, 1 - a * Y ** 2, Y], [-X, -Y, 1 - a * (X ** 2 + Y ** 2)]])
    PN = np.dot(temp, R3_s)


    # Create the transformation matrix from PQW to ijk
    # Q matrix
    Q_temp = np.dot(PN, R)
    Q = np.dot(Q_temp, W)

    # Transforms the reference frames
    r_temp1 = np.dot(R3_s, R2_xp)
    r_temp2 = np.dot(r_temp1, R1_yp)
    r_TIRS = np.dot(r_temp2, r_ITRF)

    r_temp1 = np.dot(W, r_ITRF)
    r_temp2 = np.dot(R, r_temp1)
    r_GCRF = np.dot(PN, r_temp2)

    v_temp1 = np.dot(W, v_ITRF) + np.cross(omega_earth, r_TIRS)
    v_temp2 = np.dot(PN, R)
    v_GCRF = np.dot(v_temp2, v_temp1)

    return r_GCRF, v_GCRF

def J2000toGCRF(r_J2000):
    """Transforms J2000 to GCRF to frame"""

    delta_alpha = 0.0146    # arcsec
    delta_alpha_rad = DMStoRAD(0,0,delta_alpha)     # [rad]
    epsilon_0 = -0.16617    # arcsec
    epsilon_0_rad = DMStoRAD(0,0,epsilon_0)
    eta_0 = -0.0068192
    eta_0_rad = DMStoRAD(0,0,eta_0)


    R1_eta = np.array([[1, 0, 0], [0, np.cos(eta_0_rad), np.sin(eta_0_rad)], [0, -np.sin(eta_0_rad), np.cos(eta_0_rad)]])
    R3_delta_alpha = np.array([[np.cos(-delta_alpha_rad), np.sin(-delta_alpha_rad), 0], [-np.sin(-delta_alpha_rad), np.cos(-delta_alpha_rad), 0], [0, 0, 1]])
    R2_epsilon = np.array([[np.cos(epsilon_0_rad), 0, -np.sin(epsilon_0_rad)], [0, 1, 0], [np.sin(epsilon_0_rad), 0, np.cos(epsilon_0_rad)]])

    B_temp = np.dot(R1_eta, R2_epsilon)
    B = np.dot(B_temp, R3_delta_alpha)

    r_J2000 = np.dot(B, r_J2000)

    return r_J2000

def MODtoGCRF(T_TT, r_MOD):
    """Transforms MOD to GCRF fram"""
    # arcsec
    xsi = 2306.2181*T_TT + 0.30188*T_TT**2 + 0.017998*T_TT**3
    theta = 2004.3109*T_TT**2 - 0.42665*T_TT**2 - 0.041833*T_TT**3
    z = 2306.2181*T_TT + 1.09468*T_TT**2 + 0.018203*T_TT**3

    # transform from arcsec deg then rad
    xsi = DMStoRAD(0, 0, xsi)
    theta = DMStoRAD(0, 0, theta)
    z = DMStoRAD(0, 0, z)

    R3_z = np.array([[np.cos(z), np.sin(z), 0], [-np.sin(z), np.cos(z), 0], [0, 0, 1]])
    R3_xsi= np.array([[np.cos(xsi), np.sin(xsi), 0], [-np.sin(xsi), np.cos(xsi), 0], [0, 0, 1]])
    R2_theta = np.array([[np.cos(-theta), 0, -np.sin(-theta)], [0, 1, 0], [np.sin(-theta), 0, np.cos(-theta)]])

    P_temp = np.dot(R3_xsi, R2_theta)
    P = np.dot(P_temp, R3_z)
    r_GCRF = np.dot(P,r_MOD)
    return r_GCRF

def GCRFtoJ2000(r_GCRF):
    """Transforms GCRF to J2000 frame"""

    delta_alpha = 0.0146    # arcsec
    delta_alpha_rad = DMStoRAD(0,0,delta_alpha)     # [rad]
    epsilon_0 = -0.16617    # arcsec
    epsilon_0_rad = DMStoRAD(0,0,epsilon_0)
    eta_0 = -0.0068192
    eta_0_rad = DMStoRAD(0,0,eta_0)


    R1_eta = np.array([[1, 0, 0], [0, np.cos(eta_0_rad), np.sin(eta_0_rad)], [0, -np.sin(eta_0_rad), np.cos(eta_0_rad)]])
    R3_delta_alpha = np.array([[np.cos(-delta_alpha_rad), np.sin(-delta_alpha_rad), 0], [-np.sin(-delta_alpha_rad), np.cos(-delta_alpha_rad), 0], [0, 0, 1]])
    R2_epsilon = np.array([[np.cos(epsilon_0_rad), 0, -np.sin(epsilon_0_rad)], [0, 1, 0], [np.sin(epsilon_0_rad), 0, np.cos(epsilon_0_rad)]])

    B_temp = np.dot(R3_delta_alpha, R2_epsilon)
    B = np.dot(B_temp, R1_eta)
    r_J2000= np.dot(B,r_GCRF)

    return r_J2000

def SUN(JD_UT1):
    """Converts JD UT1 to sun position vector in GCRF frame"""
    AUtokm = 149597870.691  # [km]
    #MJD_UT1 = JD_UT1 - 2400000.5
    T_UT1 = (JD_UT1 - 2451545)/36525   # in MJD
    JD_TT = JD_UT1 + 37/(3600*24) + 32.184/(3600*24)      # Terrestrial time
    #MJD_TT = JD_TT - 2400000.5
    T_TT = (JD_TT - 2451545)/36525

    # assumption
    T_TDB = T_UT1
    lambda_M = (280.460 + 36000.771*T_UT1)  # mean ecliptic longitude of the sun
    M = (np.pi/180)*(357.52772333 + 35999.0534 * T_TDB)  # mean anomaly for the sun [rad]
    lamda_ecliptic = (np.pi/180)*(lambda_M + 1.91466471*np.sin(M) + 0.019994643*np.sin(2*M))  # [rad]
    r = 1.000140612 - 0.016708617*np.cos(M) - 0.000139580*np.cos(2*M)
    epsilon = (np.pi/180)*(23.439291 - 0.0130042*T_TDB)  # Obliquity of the ecliptic  [rad]
    #epsilon = 23.439279 - 0.0130102 * T_TT - (5.086 * 10 ** -8) * T_TT ** 2 + (5.565 * 10 ** -7) * T_TT ** 3 + (
     #           1.6 * 10 ** -10) * T_TT ** 4 \
    #          + (1.21 * 10 ** -11) * T_TT ** 5  # deg!
    epsilon = epsilon * np.pi / 180  # rad
    phi_ecliptic = 0
    r_TOD = [r * np.cos(lamda_ecliptic), r*np.cos(epsilon)*np.sin(lamda_ecliptic), r*np.sin(epsilon)*np.sin(lamda_ecliptic)]
    r_MOD = r_TOD

    r_GCRF = MODtoGCRF(T_TT, r_MOD) * AUtokm     # in [km]

    return r_GCRF

def PlanetRV(JD_UTC, delta_AT, mu_earth):
    """Returns the planets position in GCRF frame in km given some inputs
    outputs earth to sun"""
    AUtokm = 149597870.691  # [km/AU]
    # assume UTC = UT1
    #JD_UT1 = JD_UTC
    #JD_TDB = JD_UTC2JD_TDB(JD_UT1, delta_AT
    # HW5 assumption
    JD_TDB = JD_UTC
    AUtokm = 149597870.691  # [km]
    T_TDB = (JD_TDB - 2451545)/36525
    T_TT = T_TDB    # assumption
    #print("T_TDB: ", T_TDB)
    # all in degrees!
    a = 1.000001018     # AU
    e = 0.01670862 - 0.000042037*T_TDB - 0.0000001236*T_TDB**2 + 0.00000000004*T_TDB**3
    i = 0.0000000 + 0.0130546*T_TDB - 0.00000931*T_TDB**2 - 0.000000034*T_TDB**3
    RAAN = 174.873174 - 0.2410908*T_TDB + 0.00004067*T_TDB**2 - 0.000001327*T_TDB**3
    omega_dash = 102.937348 + 0.3225557*T_TDB + 0.00015026*T_TDB**2 + 0.000000478*T_TDB**3
    lambda_M = 100.466449 + 35999.3728519*T_TDB - 0.00000568*T_TDB**2 + 0.000000000*T_TDB**3
    M = lambda_M - omega_dash
    omega = omega_dash - RAAN
    p = a * (1 - e ** 2)    # AU

    # Convert to radians and km
    #a = a * AUtokm
    i = i * np.pi/180
    RAAN = RAAN * np.pi/180
    omega = omega * np.pi/180
    M = M * np.pi/180
    theta = KepEqtnTrueAnomaly(M, e)    # rad!
    COEs = [a, e, i, RAAN, omega, theta]
    r_IJK, v_IJK = COEstoRV(COEs, mu_earth)
    v_IJK = v_IJK / 86400        # converts AU/day to AU/TU (divides by one solar day)
    #epsilon = 23.439279 - 0.0130102*T_TT - (5.086*10**-8) * T_TT**2 + (5.565*10**-7)*T_TT**3 + (1.6*10**-10)*T_TT**4 \
    #         + (1.21*10**-11)*T_TT**5  # deg!
    #epsilon = (np.pi / 180) * epsilon
    epsilon = (np.pi / 180) * (23.439291 - 0.0130042 * T_TDB)  # Obliquity of the ecliptic  [rad]
    R1_epsilon = np.array([[1, 0, 0], [0, np.cos(-epsilon), np.sin(-epsilon)], [0, -np.sin(-epsilon), np.cos(-epsilon)]])

    # convert from XYZ (IJK) frame to J2000
    r_J2000 = np.dot(R1_epsilon, r_IJK)
    v_J2000 = np.dot(R1_epsilon, v_IJK)
    #print("r_J2000: ", r_J2000)
    #print("v_J2000: ", v_J2000)
    r_GCRF = J2000toGCRF(r_J2000) * AUtokm
    v_GCRF = J2000toGCRF(v_J2000) * AUtokm

    return r_GCRF, v_GCRF

def KepEqtnTrueAnomaly(M, e):
    """Uses Keplers equation to calculate the true anomaly from M and e"""
    error = 1
    tol = 10**-10
    counter = 0
    MAX = 100000
    if -np.pi < M < 0 or M > np.pi:
        E = M - e/2
    else:
        E = M + e/2
    while error > tol:
        if counter == MAX:
            break
        f = M - E + e * np.sin(E)
        f_prime = 1 - e * np.cos(E)
        E_next = E + f / f_prime
        error = np.abs(E_next - E)
        E = E_next
        counter += 1
    theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    if theta < 0:
       theta = 2*np.pi + theta
    return theta

def shadow_model(r_sat, R_earth, R_sun, r_sun_pos):

    r_mag = np.linalg.norm(r_sat)
    #print("r_mag: ", r_mag)
    r_sun_sat = r_sat - r_sun_pos
    #print("r_sun_sat", r_sun_sat)
    r_sun_sat_mag = np.linalg.norm(r_sun_sat)
    #print("r_sun_sat_mag", r_sun_sat_mag)
    a = np.arcsin((R_sun/r_sun_sat_mag))                # apperhent size of the sun
    b = np.arccos((R_earth/r_mag))
    c = np.arccos((np.dot(r_sat, r_sun_sat)) / (r_mag * r_sun_sat_mag))
    #print("a: ", a)
    #print("b: ", b)
    #print("c: ", c)

    if c < np.abs(a - b):
        #print("np.abs(a - b): ", np.abs(a - b))
        gamma = 0
    elif (a + b) <= c:
        #print("a+b: ", a+b)
        gamma = 1
    else:
        #x = (c**2 + a**2 - b**2)/2
        #y = np.sqrt(a**2 - x**2)
        #A = (a ** 2) * np.arccos(x / a) + (b ** 2) * np.arccos((c-x) / b) - c * y
        #print("x: ", x)
        #print("y: ", y)
        #print("A: ", A)
        #gamma = 1 - A/(np.pi * (a**2))
        gamma = 0
    #print("gamma: ", gamma)
    return gamma

def cylindrical_shadow_model(r_sat, R_earth, r_sun_pos):
    """Returns gamma (0 = in shadow, 1 = in sun light)"""
    r_sun_pos_mag = np.linalg.norm(r_sun_pos)
    r_sat_mag = np.linalg.norm(r_sat)
    r_sun_pos_hat = r_sun_pos / r_sat_mag

    check = - np.sqrt(r_sat_mag ** 2 - R_earth ** 2)
    dot = np.dot(r_sat, r_sun_pos_hat)
    if dot < check:
        gamma = 0
    else:
        gamma = 1

    return gamma

def aspherical_accl(r_ITRF, J2, J3, mu_earth, R_earth):
    """Takes position in ITRF frame and outputs the aspherical acceleration due to J2 and J3"""
    r_abs = np.linalg.norm(r_ITRF)

    # J2
    a_J2_I = ((-3 * J2 * mu_earth * (R_earth ** 2) * r_ITRF[0]) / (2 * (r_abs ** 5))) \
             * (1 - 5 * (r_ITRF[2] ** 2) / (r_abs ** 2))
    a_J2_J = ((-3 * J2 * mu_earth * (R_earth ** 2) * r_ITRF[1]) / (2 * (r_abs ** 5))) \
             * (1 - 5 * (r_ITRF[2] ** 2) / (r_abs ** 2))
    a_J2_K = ((-3 * J2 * mu_earth * (R_earth ** 2) * r_ITRF[2]) / (2 * (r_abs ** 5))) \
             * (3 - 5 * (r_ITRF[2] ** 2) / (r_abs ** 2))
    a_J2 = np.array([a_J2_I, a_J2_J, a_J2_K])


    # J3
    a_J3_I = ((-5 * J3 * mu_earth * (R_earth ** 3) * r_ITRF[0]) / (2 * (r_abs ** 7))) \
             * (3 * r_ITRF[2] - 7 * (r_ITRF[2] ** 3) / (r_abs ** 2))
    a_J3_J = ((-5 * J3 * mu_earth * (R_earth ** 3) * r_ITRF[1]) / (2 * (r_abs ** 7))) \
             * (3 * r_ITRF[2] - 7 * (r_ITRF[2] ** 3) / (r_abs ** 2))
    a_J3_K = ((-5 * J3 * mu_earth * (R_earth ** 3)) / (2 * (r_abs ** 7))) \
             * (6 * (r_ITRF[2] ** 2) - 7 * (r_ITRF[2] ** 4) / (r_abs ** 2) - (3 / 5) * (r_abs ** 2))
    a_J3 = np.array([a_J3_I, a_J3_J, a_J3_K])

    a_p = a_J2 + a_J3

    return a_J2, a_J3

def SRP(r_sun, r_sat, C_R, A_div_m, R_earth, R_sun, r_sun_pos):
    """returns the solar radiation pressure in km/s^2 given the suns position"""
    r_sat_sun = r_sun - r_sat
    r_sat_sun_abs = np.linalg.norm(r_sat_sun)
    # Checks if in shadow of the Earth or not
    gamma = cylindrical_shadow_model(r_sat, R_earth, r_sun_pos)

    # solar pressure
    P_srp = 4.57 * 10 ** -6                                             # [N/m^2]
    a_SRP = - gamma * P_srp * C_R * A_div_m * r_sat_sun/r_sat_sun_abs   # [m/s^2]
    a_SRP = a_SRP / 1000                                                # [km/s^2]
    return a_SRP

def atmospheric_density(h_ellp):
    """returns the density of the atmosphere based on the U.S standard Atmosphere model (1976)"""
    # reference density rho_0
    # reference altitude h_0
    # actual altitude above ellipsoid h_ellp
    # scale height H
    #print("h_ellp: ", h_ellp)
    if h_ellp >= 0  and h_ellp < 25 :
        h_0 = 0
        rho_0 = 1.225
        H = 7.249
    if h_ellp >= 25 and h_ellp < 30 :
        h_0 = 25
        rho_0 = 3.899*10**(-2)
        H = 6.349
    if h_ellp >= 30 and h_ellp < 40 :
        h_0 = 30
        rho_0 = 1.774*10**(-2)
        H = 6.682
    if h_ellp >= 40 and h_ellp < 50 :
        h_0 = 40
        rho_0 = 3.972*10**(-3)
        H = 7.554
    if h_ellp >= 50 and h_ellp < 60 :
        h_0 = 50
        rho_0 = 1.057*10**(-3)
        H = 8.382
    if h_ellp >= 60 and h_ellp < 70 :
        h_0 = 60
        rho_0 = 3.206*10**(-4)
        H = 7.714
    if h_ellp >= 70 and h_ellp < 80 :
        h_0 = 70
        rho_0 = 8.77*10**(-5)
        H = 6.549
    if h_ellp >= 80 and h_ellp < 90 :
        h_0 = 80
        rho_0 = 1.905*10**(-5)
        H = 5.799
    if h_ellp >= 90 and h_ellp < 100 :
        h_0 = 90
        rho_0 = 3.396*10**(-6)
        H = 5.382
    if h_ellp >= 100 and h_ellp < 110 :
        h_0 = 100
        rho_0 = 5.297*10**(-7)
        H = 5.877
    if h_ellp >= 110 and h_ellp < 120 :
        h_0 = 110
        rho_0 = 9.661*10**(-8)
        H = 7.263
    if h_ellp >= 120 and h_ellp < 130 :
        h_0 = 120
        rho_0 = 2.438*10**(-8)
        H = 9.473
    if h_ellp >= 130 and h_ellp < 140 :
        h_0 = 130
        rho_0 = 8.484*10**(-9)
        H = 12.636
    if h_ellp >= 140 and h_ellp < 150 :
        h_0 = 140
        rho_0 = 3.845*10**(-9)
        H = 16.149
    if h_ellp >= 150 and h_ellp < 180 :
        h_0 = 150
        rho_0 = 2.070*10**(-9)
        H = 22.523
    if h_ellp >= 180 and h_ellp < 200 :
        h_0 = 180
        rho_0 = 5.464*10**(-10)
        H = 29.74
    if h_ellp >= 200 and h_ellp < 250 :
        h_0 = 200
        rho_0 = 2.789*10**(-10)
        H = 37.105
    if h_ellp >= 250 and h_ellp < 300 :
        h_0 = 250
        rho_0 = 7.248*10**(-11)
        H = 45.546
    if h_ellp >= 300 and h_ellp < 350 :
        h_0 = 300
        rho_0 = 2.418*10**(-11)
        H = 53.628
    if h_ellp >= 350 and h_ellp < 400 :
        h_0 = 350
        rho_0 = 9.518*10**(-12)
        H = 53.298
    if h_ellp >= 400 and h_ellp < 450 :
        h_0 = 400
        rho_0 = 3.725*10**(-12)
        H = 58.515
    if h_ellp >= 450 and h_ellp < 500 :
        h_0 = 450
        rho_0 = 1.585*10**(-12)
        H = 60.828
    if h_ellp >= 500 and h_ellp < 600 :
        h_0 = 500
        rho_0 = 6.967*10**(-13)
        H = 63.822
    if h_ellp >= 600 and h_ellp < 700 :
        h_0 = 600
        rho_0 = 1.454*10**(-13)
        H = 71.835
    if h_ellp >= 700 and h_ellp < 800:
        h_0 = 700
        rho_0 = 3.614*10**(-14)
        H = 88.667
    if h_ellp >= 800 and h_ellp < 900 :
        h_0 = 800
        rho_0 = 1.17*10**(-14)
        H = 124.64
    if h_ellp >= 900 and h_ellp < 1000 :
        h_0 = 900
        rho_0 = 5.245*10**(-15)
        H = 181.05
    if h_ellp >= 1000:
        h_0 = 1000
        rho_0 = 3.019*10**(-15)
        H = 268
    if h_ellp < 0:
        h_0 = 0
        rho_0 = 1.225
        H = 7.249
        raise ValueError("below ground!")

    rho = rho_0 * np.exp((- h_ellp - h_0)/H) * (1000 ** 3)       # [kg/km^3]
    return rho

def propogate_orbit_ode_all_pertrubations(COEs, T, mu_earth, R_earth, w_earth, R_sun, mu_moon, mu_sun, A_div_m, C_D, C_R,
                                          JD_UT1_i, delta_AT, J2, J3, PERTB, special_case):
    """propogates the orbit given COEs and time vector"""
    tp = 0  # time of perihelion
    tol = 10**-10  # tolerance
    # arrays
    COEs_arr = []
    R_arr = []
    V_arr = []

    # Saves initial positions
    COEs_arr.append(COEs)
    r, v = COEstoRV(COEs, mu_earth)
    R_arr.append(r)
    V_arr.append(v)
    r_transp = np.asarray(np.transpose(r))
    v_transp = np.asarray(np.transpose(v))
    Y = np.append(r_transp, v_transp)

    # Does the integrations
    Y_solv = integrate.odeint(function_perturbation_all_pertrubations, Y, T, args=(mu_earth, R_earth, w_earth, R_sun, mu_moon,
                        mu_sun, A_div_m, C_D, C_R, JD_UT1_i, delta_AT, J2, J3, PERTB, special_case), rtol=tol, atol=tol)

    Y_solv = np.array(Y_solv)
    return Y_solv

def function_perturbation_all_pertrubations(Y, t, mu_earth, R_earth, w_earth, R_sun, mu_moon, mu_sun, A_div_m, C_D, C_R,
                                            JD_UT1_i, delta_AT, J2, J3, PERTB, special_case):
    """Defines the ordinary differential equation/function of the two-body problem"""

    # r & v are in the GCRF frame!
    r = np.asarray(Y[0:3])
    v = np.asarray(Y[3:])
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    #print("r: ", r)
    #print("v: ", v)
    ## 2-body acceleration due to Earth
    a_earth = - mu_earth * r/(r_mag**3)     # [km/s^2]

    ##  pertrubation due to the Sun and Moon (third body)
    # Calculates the Suns position
    #print("t: ", t)
    JD_UT1 = JD_UT1_i + t / 86400         # updates JD_UT1 to current time in JD [sec]
    #r_sun_GCRF = SUN(JD_UT1)            # returns the suns position from Earth in [km]
    r_sun_GCRF, v_sun_GCRF = PlanetRV(JD_UT1, delta_AT, mu_earth)
    a_3rd_body = third_body_pertubation(r, r_sun_GCRF, mu_sun, mu_moon, JD_UT1, delta_AT)   # [km/s^2]

    ## pertrubation due to J2 & J3
    a_J2, a_J3 = aspherical_accl(r, J2, J3, mu_earth, R_earth)  # [km/s^2]

    ## pertrubation due to atmospheric drag
    h_ellp = r_mag - R_earth
    rho = atmospheric_density(h_ellp)
    v_rel = v - np.cross(w_earth, r)
    a_atm = - 0.5 * C_D * A_div_m * rho * np.dot(v_rel, v_rel) * (v_rel / (np.linalg.norm(v_rel)))     # [km/s^2]

    ## pertrubation due to solar radiation pressure
    a_SRP = SRP(r_sun_GCRF, r, C_R, A_div_m, R_earth, R_sun, r_sun_GCRF)    # [km/s^2]

    if PERTB is "2_body":
        a_tot = a_earth
    if PERTB is "drag":
        a_tot = a_earth + a_atm
    if PERTB is "J2":
        a_tot = a_earth + a_J2
    if PERTB is "J3":
        a_tot = a_earth + a_J3
    if PERTB is "J2_J3":
        a_tot = a_earth + a_J2 + a_J3
    if PERTB is "SRP":
        a_tot = a_earth + a_SRP
    if PERTB is "3rd_body":
        a_tot = a_earth + a_3rd_body
    if PERTB is "all":
        a_tot = a_earth + a_J2 + a_J3 + a_atm + a_SRP + a_3rd_body
    """
    print("a_2body: ", a_earth)
    print("a_atm: ", a_atm)
    print("a_SRP: ", a_SRP)
    print("a_3rd: ", a_3rd_body)
    print("a_J2: ", a_J2)
    print("a_J3: ", a_J3)
    print("a_all: ", a_tot)
    print("")
    """

    #######
    """"""
    i_start = 98.540 * D2R_  # [rad] - Inclination (start!)
    tol = 0.01
    # convert RV to COE
    COE = RVtoCOE_special(r, v, mu_earth, special_case)
    a = COE[0]
    e = COE[1]
    i = COE[2]
    RAAN = COE[3]
    omega = COE[4]
    u = COE[5]
    di = i - i_start
    dRAAN = RAAN - 0
    #print("dRAAN : ", dRAAN)
    #print("di: ", di)
    if np.abs(di) > tol:
        print("Boost!")
        print("di: ", di)
        n = np.sqrt(mu_earth / (a ** 3))       # mean motion
        dv_w = - 2 * di * n * a / np.cos(u)   # check radians!
        dv_RSW = np.array([0, 0, dv_w])
        # convert dv_w (RSW) to dv_k (IJK)
        # add dv_k
        R = r / np.linalg.norm(r)
        W = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
        S = np.cross(W, R)
        Q_RSW_2_IJK = np.array([R, S, W])
        dv_IJK = np.dot(Q_RSW_2_IJK, dv_RSW)
        print("dv_IJK: ", dv_IJK)
    else:
        dv_IJK = np.array([0, 0, 0])

    #dv_IJK = np.array([0, 0, 0])
    #######
    Y = [v + dv_IJK, a_tot]
    Y = np.append(Y[0], Y[1])
    return np.asarray(Y)

def third_body_pertubation(r, r_sun_GCRF, mu_sun, mu_moon, JD_UT1, delta_AT):
    """returns the pertubation accelerations from the moon and the sun"""
    # Time
    JD_TDB = JD_UTC2JD_TDB(JD_UT1, delta_AT)

    # Position vectors
    r_earth_moon = getMoonPos_wrt_Earth(JD_TDB)
    r_sat_moon = r_earth_moon - r
    r_earth_moon_mag = np.linalg.norm(r_earth_moon)
    r_sat_moon_mag = np.linalg.norm(r_sat_moon)
    r_sat_sun = r_sun_GCRF - r
    mag_r_sun_GCRF = np.linalg.norm(r_sun_GCRF)
    mag_r_sat_sun = np.linalg.norm(r_sat_sun)

    # Calculates the perturbations
    a_3rd_body_sun = mu_sun * (r_sat_sun / (mag_r_sat_sun ** 3) - r_sun_GCRF / (mag_r_sun_GCRF ** 3))   # Acceleration from satellite to Earth due to the Sun
    a_3rd_body_moon = mu_moon * (r_sat_moon / (r_sat_moon_mag ** 3) - r_earth_moon / (r_earth_moon_mag ** 3))   # Acceleration from satellite to Earth due to the Moon
    a_3rd_bodies = a_3rd_body_sun + a_3rd_body_moon
    return a_3rd_bodies

def CW_equations(rho_i, rho_dot_i, n, t):
    """Returns the relative future position and velocity using the CW equations"""
    THETA_pp = np.array([[4 - 3 * np.cos(n * t), 0, 0],
                        [6 * (np.sin(n * t) - n * t), 1, 0],
                        [0, 0, np.cos(n * t)]])
    THETA_p_p_dot = np.array([[(1 / n) * np.sin(n * t), (2 / n) * (1 - np.cos(n * t)), 0],
                              [(2 / n) * (np.cos(n * t) - 1), (1 / n) * (4 * np.sin(n * t) - 3 * n * t), 0],
                              [0, 0, (1 / n) * np.sin(n * t)]])
    THETA_p_dot_p = np.array([[3 * n * np.sin(n * t), 0, 0],
                        [6 * n * (np.cos(n * t) - 1), 0, 0],
                        [0, 0, -n * np.sin(n * t)]])
    THETA_p_dot_p_dot = np.array([[np.cos(n * t), 2 * np.sin(n * t), 0],
                        [-2 * np.sin(n * t), 4 * np.cos(n * t) - 3, 0],
                        [0, 0, np.cos(n * t)]])
    rho = np.dot(THETA_pp, rho_i) + np.dot(THETA_p_p_dot, rho_dot_i)
    rho_dot = np.dot(THETA_p_dot_p, rho_i) + np.dot(THETA_p_dot_p_dot, rho_dot_i)

    return rho, rho_dot

def getMoonPos_wrt_Earth(JD_TDB):
    '''
    Get the position of the Moon with respect to the Earth based on the
    2010 Astronomical Almanac algorithms.  Returns position vector in
    J2000, geocentric, equatorial frame.
    '''
    #  Pre-compute the conversion from degrees to radians
    D2R_ = np.pi / 180.0
    #  Convenience functions to make copy/paste from MATLAB easier
    def sind(arg):
        return np.sin(arg * D2R_)

    def cosd(arg):
        return np.cos(arg * D2R_)

    # Convert the input time into Julian Centuries
    T_TDB = (JD_TDB - 2451545.0) / 36525.0

    long_ecliptic = 218.32 + 481267.8813 * T_TDB \
                    + 6.29 * sind(134.9 + 477198.85 * T_TDB) \
                    - 1.27 * sind(259.2 - 413335.38 * T_TDB) \
                    + 0.66 * sind(235.7 + 890534.23 * T_TDB) \
                    + 0.21 * sind(269.9 + 954397.70 * T_TDB) \
                    - 0.19 * sind(357.5 + 35999.05 * T_TDB) \
                    - 0.11 * sind(186.6 + 966404.05 * T_TDB)  # deg

    lat_ecliptic = 5.13 * sind(93.3 + 483202.03 * T_TDB) \
                   + 0.28 * sind(228.2 + 960400.87 * T_TDB) \
                   - 0.28 * sind(318.3 + 6003.18 * T_TDB) \
                   - 0.17 * sind(217.6 - 407332.20 * T_TDB)  # deg

    parralax = 0.9508 + 0.0518 * cosd(134.9 + 477198.85 * T_TDB) \
               + 0.0095 * cosd(259.2 - 413335.38 * T_TDB) \
               + 0.0078 * cosd(235.7 + 890534.23 * T_TDB) \
               + 0.0028 * cosd(269.9 + 954397.70 * T_TDB)  # deg

    #  Use the unwrap() function to get an angle in the range [-pi,pi]
    long_ecliptic = np.unwrap([long_ecliptic * D2R_])[0]
    lat_ecliptic = np.unwrap([lat_ecliptic * D2R_])[0]
    parralax = np.unwrap([parralax * D2R_])[0]

    obliquity = 23.439291 - 0.0130042 * T_TDB  # deg
    obliquity = obliquity * D2R_

    # Pre-compute cosine and sine values that will be needed at least twice
    cos_lag_ecl = np.cos(lat_ecliptic)
    sin_lon_ecl = np.sin(long_ecliptic)
    sin_lat_ecl = np.sin(lat_ecliptic)
    cos_obliq = np.cos(obliquity)
    sin_obliq = np.sin(obliquity)

    # ------------- calculate moon position vector ----------------
    pos_j2000 = np.zeros((3,))
    pos_j2000[0] = cos_lag_ecl * np.cos(long_ecliptic)
    pos_j2000[1] = cos_obliq * cos_lag_ecl * sin_lon_ecl - sin_obliq * sin_lat_ecl
    pos_j2000[2] = sin_obliq * cos_lag_ecl * sin_lon_ecl + cos_obliq * sin_lat_ecl

    #  Generate position in km where Re = 6378.1363 km
    pos_j2000 *= 6378.1363 / np.sin(parralax)
    pos_GCRF = J2000toGCRF(pos_j2000)
    #  Return position
    return pos_GCRF

def JD_UTC2JD_TDB(JD_UT1, delta_AT):
    """Converts UT1 to TDB"""
    JD_TT = JD_UT1 + delta_AT / 86400 + 32.184 / 86400        #  starting in Epoch J2000 [days]
    g = 357.53 + 0.98560028 * (JD_TT - 2451545)
    L_L_J = 246.11 + 0.90251792 * (JD_TT - 2451545)
    delta_TDB = 0.001657 * np.sin(g * np.pi / 180.0) + 0.000022 * np.sin(L_L_J * np.pi / 180.0)
    JD_TDB = JD_TT + delta_TDB
    return JD_TDB

def leap_sec(year):
    """Returns leap second (dAT)"""
    year = int(year)
    if year == 1991 or year < 1991:
        leap_sec = 26
    elif year == 1992:
        leap_sec = 27
    elif year == 1993:
        leap_sec = 28
    elif year == 1994 or year == 1995:
        leap_sec = 29
    elif year == 1996:
        leap_sec = 30
    elif year == 1997 or year == 1998:
        leap_sec = 31
    elif 1999 <= year < 2006:
        leap_sec = 32
    elif 2006 <= year < 2009:
        leap_sec = 33
    elif 2009 <= year < 2012:
        leap_sec = 34
    elif 2012 <= year < 2015:
        leap_sec = 35
    elif 2012 <= year < 2015:
        leap_sec = 35
    elif 2015 <= year < 2017:
        leap_sec = 36
    elif year >= 2017:
        leap_sec = 37
    else:
        raise ValueError("Year is not an integer or negative!")
    dAT = leap_sec
    return dAT

def theta_GMST(T_UT1):
    """Returns theta_GMST"""
    theta_GMST_sec = 67310.54841 + (876600 * 3600 + 8640184.812866) * T_UT1 + 0.093104 * (T_UT1 ** 2) - \
                     (6.2 * 10 ** - 6) * T_UT1 ** 3  # [s]
    theta_GMST_day = theta_GMST_sec / 86400 - int(theta_GMST_sec / 86400)  # [days]
    theta_GMST_rad = HMStoRAD(theta_GMST_day * 24, 0, 0)
    return theta_GMST_rad

def ITRFtoGCRF(r_ITRF, v_ITRF, date, UTC, dUT1, x_p, y_p, dX, dY):
    """Transforms ITRF frame vectors to GCRF frame, send angles in rad!"""
    # UT1, TAI, GPS, TT = CONVTIME(UTC, dUT1, dAT)

    dAT = leap_sec(date[0])  # [sec]
    JD_UTC = UTCtoJD(date, UTC)
    JD_UT1 = JD_UTC + dUT1 / 86400
    JD_TAI = JD_UT1 + dAT / 86400
    JD_TT = JD_TAI + 32.184 / 86400
    T_TT = (JD_TT - 2451545) / 36525

    # Polynomials
    X = -0.016671 + 2004.191898 * T_TT - 0.4297829 * T_TT ** 2 - 0.19861834 * T_TT ** 3 + 0.000007578 * T_TT ** 4 + 0.0000059285 * T_TT ** 5  # arcsec!
    Y = - 0.006951 - 0.025896 * T_TT - 22.4072747 * T_TT ** 2 + 0.00190059 * T_TT ** 3 + 0.001112526 * T_TT ** 4 + 0.0000001358 * T_TT ** 5  # arcsec!
    s = - 0.5 * X * Y + 0.000094 + 0.00380865 * T_TT - 0.00012268 * T_TT ** 2 - 0.07257411 * T_TT ** 3 + 0.00002798 * T_TT ** 4 + 0.00001562 * T_TT ** 5  # arcsec!
    X = X * Arcs2Rad_
    Y = Y * Arcs2Rad_
    s = s * Arcs2Rad_  # rad
    X = X + dX
    Y = Y + dY
    a = 0.5 + 0.125 * (X ** 2 + Y ** 2)  # arcsec
    a = a * Arcs2Rad_  # rad
    s_prim = 0.000047 * T_TT  # arcsec
    s_prim = s_prim * Arcs2Rad_  # rad

    # Earth rotation in TIRS frame
    LOD = dUT1
    omega_earth = [0, 0, 7.292115146706979 * 10 ** -5 * (1 - LOD / 86400)]  # TIRS frame

    # Earths rotation angle
    theta_ERA = 2 * np.pi * (0.779057273264 + 1.00273781191135448 * (JD_UT1 - 2451545))  # [rad]
    theta_ERA = theta_ERA - 2 * np.pi * int(theta_ERA / (2 * np.pi))

    # W matrix
    R3_s_prim = np.array([[np.cos(-s_prim), np.sin(-s_prim), 0], [-np.sin(-s_prim), np.cos(-s_prim), 0], [0, 0, 1]])
    R1_yp = np.array([[1, 0, 0], [0, np.cos(y_p), np.sin(y_p)], [0, -np.sin(y_p), np.cos(y_p)]])
    R2_xp = np.array([[np.cos(x_p), 0, -np.sin(x_p)], [0, 1, 0], [np.sin(x_p), 0, np.cos(x_p)]])
    W_temp = np.dot(R3_s_prim, R2_xp)
    W = np.dot(W_temp, R1_yp)

    # R matrix
    R = np.array([[np.cos(-theta_ERA), np.sin(-theta_ERA), 0], [-np.sin(-theta_ERA), np.cos(-theta_ERA), 0],
                  [0, 0, 1]])  # R3 rotation

    # PN matrix
    R3_s = np.array([[np.cos(s), np.sin(s), 0], [-np.sin(s), np.cos(s), 0], [0, 0, 1]])
    temp = np.array(
        [[1 - a * (X ** 2), -a * X * Y, X], [-a * X * Y, 1 - a * (Y ** 2), Y], [-X, -Y, 1 - a * (X ** 2 + Y ** 2)]])
    PN = np.dot(temp, R3_s)

    # Create the transformation matrix from ITRF to GCRF (or vice versa)
    Q_temp = np.dot(PN, R)
    Q_ITRFtoGCRF = np.dot(Q_temp, W)

    temp_inv = np.dot(np.transpose(W), np.transpose(R))
    Q_GCRFtoITRF = np.dot(temp_inv, np.transpose(PN))

    # Transforms the reference frames
    # position
    r_GCRF = np.dot(Q_ITRFtoGCRF, r_ITRF)
    # velocity
    r_temp1 = np.dot(R3_s, R2_xp)
    r_temp2 = np.dot(r_temp1, R1_yp)
    r_TIRS = np.dot(r_temp2, r_ITRF)
    v_temp1 = np.dot(W, v_ITRF) + np.cross(omega_earth, r_TIRS)
    v_temp2 = np.dot(PN, R)
    v_GCRF = np.dot(v_temp2, v_temp1)

    return r_GCRF, v_GCRF

def GCRFtoITRF(r_GCRF, v_GCRF, JD_UTC, dUT1, dAT, x_p, y_p, dX, dY):
    """Transforms ITRF frame vectors to GCRF frame, send angles in rad!"""
    # Converts time
    JD_UT1 = JD_UTC + dUT1 / 86400
    JD_TAI = JD_UT1 + dAT / 86400
    JD_TT = JD_TAI + 32.184 / 86400
    T_TT = (JD_TT - 2451545) / 36525

    # Polynomials
    X = -0.016671 + 2004.191898 * T_TT - 0.4297829 * T_TT ** 2 - 0.19861834 * T_TT ** 3 + 0.000007578 * T_TT ** 4 + 0.0000059285 * T_TT ** 5  # arcsec!
    Y = - 0.006951 - 0.025896 * T_TT - 22.4072747 * T_TT ** 2 + 0.00190059 * T_TT ** 3 + 0.001112526 * T_TT ** 4 + 0.0000001358 * T_TT ** 5  # arcsec!
    s = - 0.5 * X * Y + 0.000094 + 0.00380865 * T_TT - 0.00012268 * T_TT ** 2 - 0.07257411 * T_TT ** 3 + 0.00002798 * T_TT ** 4 + 0.00001562 * T_TT ** 5  # arcsec!
    X = X * Arcs2Rad_
    Y = Y * Arcs2Rad_
    s = s * Arcs2Rad_  # rad
    X = X + dX
    Y = Y + dY
    a = 0.5 + 0.125 * (X ** 2 + Y ** 2)  # arcsec
    a = a * Arcs2Rad_  # rad
    s_prim = 0.000047 * T_TT  # arcsec
    s_prim = s_prim * Arcs2Rad_  # rad

    # Earth rotation in TIRS frame
    LOD = dUT1
    omega_earth = [0, 0, 7.292115146706979 * 10 ** -5 * (1 - LOD / 86400)]  # TIRS frame

    # Earths rotation angle
    theta_ERA = 2 * np.pi * (0.779057273264 + 1.00273781191135448 * (JD_UT1 - 2451545))  # [rad]
    theta_ERA = theta_ERA - 2 * np.pi * int(theta_ERA / (2 * np.pi))

    # W matrix
    R3_s_prim = np.array([[np.cos(-s_prim), np.sin(-s_prim), 0], [-np.sin(-s_prim), np.cos(-s_prim), 0], [0, 0, 1]])
    R1_yp = np.array([[1, 0, 0], [0, np.cos(y_p), np.sin(y_p)], [0, -np.sin(y_p), np.cos(y_p)]])
    R2_xp = np.array([[np.cos(x_p), 0, -np.sin(x_p)], [0, 1, 0], [np.sin(x_p), 0, np.cos(x_p)]])
    W_temp = np.dot(R3_s_prim, R2_xp)
    W = np.dot(W_temp, R1_yp)

    # R matrix
    R = np.array([[np.cos(-theta_ERA), np.sin(-theta_ERA), 0], [-np.sin(-theta_ERA), np.cos(-theta_ERA), 0],
                  [0, 0, 1]])  # R3 rotation

    # PN matrix
    R3_s = np.array([[np.cos(s), np.sin(s), 0], [-np.sin(s), np.cos(s), 0], [0, 0, 1]])
    temp = np.array(
        [[1 - a * (X ** 2), -a * X * Y, X], [-a * X * Y, 1 - a * (Y ** 2), Y], [-X, -Y, 1 - a * (X ** 2 + Y ** 2)]])
    PN = np.dot(temp, R3_s)

    # Create the transformation matrix from ITRF to GCRF (or vice versa)
    Q_temp = np.dot(PN, R)
    Q_ITRFtoGCRF = np.dot(Q_temp, W)

    temp_inv = np.dot(np.transpose(W), np.transpose(R))
    Q_GCRFtoITRF = np.dot(temp_inv, np.transpose(PN))

    # Transforms the reference frames
    # position
    r_ITRF = np.dot(Q_GCRFtoITRF, r_GCRF)

    # velocity
    r_temp1 = np.dot(R3_s, R2_xp)
    r_temp2 = np.dot(r_temp1, R1_yp)
    r_TIRS = np.dot(r_temp2, r_ITRF)
    v_temp1 = np.dot(np.transpose(PN), v_GCRF) - np.cross(omega_earth, r_TIRS)
    v_temp2 = np.dot(np.transpose(R), v_temp1)
    v_ITRF = np.dot(np.transpose(W), v_temp2)

    return r_ITRF, v_ITRF

def ECEFtoLatLon(r_ECEF, e_earth, R_earth):
    """Returns geodic latitude, longitude (in radians) and ellipsoidal height (in km)"""
    r_I = r_ECEF[0]
    r_J = r_ECEF[1]
    r_K = r_ECEF[2]
    r_delta_sat = np.sqrt(r_I ** 2 + r_J ** 2)    # [km]
    alpha_sin = np.arcsin(r_J / r_delta_sat)      # [rad]
    alpha_cos = np.arccos(r_I / r_delta_sat)      # [rad]
    alpha = np.abs(alpha_sin)
    #print("r_I: ", r_I)
    #print("r_J: ", r_J)
    # quadrant check
    if r_I >= 0:
        # 1st quadrant
        alpha = alpha_sin
    elif r_J > 0:
        # 2nd quadrant
        alpha = alpha_cos
    elif r_I < 0 and r_J < 0:
        # 3rd quadrant
        alpha = np.arctan(r_J / r_I) - np.pi
    longitude = alpha                       # [rad]
    #print("lon: ", longitude)
    r_delta = r_delta_sat
    delta = np.arctan(r_K / r_delta_sat)
    lat_gd = 0                              # [rad]
    lat_gd_next = delta                     # [rad] start value
    S_earth = R_earth * (1 - e_earth ** 2) / (np.sqrt(1 - (e_earth ** 2) * np.sin(lat_gd)))
    C_earth = S_earth / (np.sqrt(1 - (e_earth ** 2) * np.sin(lat_gd)))

    tol = 10 ** - 6     # tolerance

    while np.abs(lat_gd_next - lat_gd) > tol:
        lat_gd = lat_gd_next
        C_earth = R_earth / (np.sqrt(1 - (e_earth ** 2) * np.sin(lat_gd)))
        temp_arg = (r_ECEF[2] + C_earth * (e_earth ** 2) * np.sin(lat_gd)) / r_delta
        lat_gd_next = np.arctan(temp_arg)
    h_ellp = r_delta / np.cos(lat_gd_next) - C_earth    # [km]

    return lat_gd, longitude, h_ellp

def propogate_orbit_ode_Moon_orbit(COEs, T, mu_prim, R_prim, R_sun, mu_sec, mu_third, A_div_m, C_R, J2,
                                          JD_UT1_i, delta_AT, PERTB, special_case):
    """propogates the orbit given COEs and time vector"""
    tp = 0  # time of perihelion
    tol = 10**-10  # tolerance

    # Saves initial positions
    r, v = COEstoRV(COEs, mu_prim)
    r_transp = np.asarray(np.transpose(r))
    v_transp = np.asarray(np.transpose(v))
    Y = np.append(r_transp, v_transp)

    # Does the integrations
    Y_solv = integrate.odeint(function_perturbation_Moon_orbit, Y, T, args=(mu_prim, R_prim, R_sun, mu_sec,
                        mu_third, A_div_m, C_R, J2, JD_UT1_i, delta_AT, PERTB, special_case), rtol=tol, atol=tol)

    Y_solv = np.array(Y_solv)
    return Y_solv

def function_perturbation_Moon_orbit(Y, t, mu_prim, R_prim, R_sun, mu_sec, mu_third, A_div_m, C_R, J2,
                                            JD_UT1_i, delta_AT, PERTB, special_case):
    """Defines the ordinary differential equation/function of the two-body problem"""

    # r & v are in the GCRF frame!
    r = np.asarray(Y[0:3])
    v = np.asarray(Y[3:])
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    #print("r: ", r)
    #print("v: ", v)
    ## 2-body acceleration due to Earth
    a_2body = - mu_prim * r/(r_mag**3)     # [km/s^2]

    ##  pertrubation due to the Sun and Moon (third body)
    # Calculates the Suns position
    #print("t: ", t)
    JD_UT1 = JD_UT1_i + t / 86400         # updates JD_UT1 to current time in JD [sec]
    #r_sun_GCRF = SUN(JD_UT1)            # returns the suns position from Earth in [km]
    r_sun_GCRF, v_sun_GCRF = PlanetRV(JD_UT1, delta_AT, mu_prim)
    a_3rd_body = third_body_pertubation_Moon(r, r_sun_GCRF, mu_sec, mu_third, JD_UT1, delta_AT)   # [km/s^2]

    ## pertrubation due to solar radiation pressure
    a_SRP = SRP(r_sun_GCRF, r, C_R, A_div_m, R_prim, R_sun, r_sun_GCRF)    # [km/s^2]

    ## pertrubation due to J2 & J3
    a_J2, a_J3 = aspherical_accl(r, J2, 0, mu_prim, R_prim)  # [km/s^2]

    if PERTB is "2_body":
        a_tot = a_2body
    if PERTB is "J2":
        a_tot = a_2body + a_J2
    if PERTB is "SRP":
        a_tot = a_2body + a_SRP
    if PERTB is "3rd_body":
        a_tot = a_2body + a_3rd_body
    if PERTB is "all":
        a_tot = a_2body + a_SRP + a_3rd_body + a_J2
    """
    print("a_2body: ", a_2body)
    print("a_SRP: ", a_SRP)
    print("a_3rd: ", a_3rd_body)
    print("a_all: ", a_tot)
    print("")
    """
    #print("\na_2body: ", a_2body)
    #a_tot = a_earth # a_3rd_body
    Y = [v, a_tot]
    Y = np.append(Y[0], Y[1])
    return np.asarray(Y)

def third_body_pertubation_Moon(r, r_sun_GCRF, mu_earth, mu_sun, JD_UT1, delta_AT):
    """returns the pertubation accelerations from the moon and the sun"""
    # Time
    JD_TDB = JD_UTC2JD_TDB(JD_UT1, delta_AT)

    ## Moon is the primary body in the origin

    # Position vectors
    r_earth_moon = getMoonPos_wrt_Earth(JD_TDB)
    # replacing earth for the moon
    r_moon_earth = - r_earth_moon
    r_moon_sun = r_sun_GCRF - r_earth_moon

    r_sat_earth = r_moon_earth - r
    r_moon_earth_mag = np.linalg.norm(r_moon_earth)
    r_sat_earth_mag = np.linalg.norm(r_sat_earth)
    r_sat_sun = r_moon_sun - r

    mag_r_moon_sun = np.linalg.norm(r_moon_sun)
    mag_r_sat_sun = np.linalg.norm(r_sat_sun)

    # Calculates the perturbations
    a_3rd_body_sun = mu_sun * (r_sat_sun / (mag_r_sat_sun ** 3) - r_moon_sun / (mag_r_moon_sun ** 3))
    a_3rd_body_earth = mu_earth * (r_sat_earth / (r_sat_earth_mag ** 3) - r_moon_earth / (r_moon_earth_mag ** 3))
    a_3rd_bodies = a_3rd_body_sun + a_3rd_body_earth
    #print("a_3rd_body_earth: ", a_3rd_body_earth)
    #print("a_3rd_body_sun: ", a_3rd_body_sun)
    return a_3rd_bodies
