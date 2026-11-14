import numpy as np 
import rebound
from resonances import angles

# Simulation's configuration
units = ('AU', 'yr', 'Msun')
integrator = 'ias15'

# Adding the Sun and giant planets
massive_bodies = ['Sun', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']


def build_initial_simulations(epochs, massive_bodies=massive_bodies, units=units):
    initial_simulations = {}
    for n, epoch in enumerate(epochs.unique(), start=1):

        date_Horizons = f"JD{epoch}"

        # Simulation configuration
        sim = rebound.Simulation()
        sim.units = units

        # Set up the planetary system at the given epoch
        for body in massive_bodies:
            sim.add(body, date=date_Horizons)
            sim.particles[-1].name = body

        # Move the frame to the center of mass of the system
        sim.move_to_com()

        # Store the initial environment for the given epoch
        initial_simulations[epoch] = sim

    print("\nPlanetary systems loaded.")
    return initial_simulations


def integrate_candidate(row, initial_sim, N=2048, units=units, integrator=integrator, all_modes = True, res_cycles=5):
    
    sim = initial_sim.copy()
    sim.integrator = integrator

    # Calculate the resonant modes
    if all_modes == True:
        modes = angles.resonant_modes(row['k_p'], row['k'])
    else:
        modes = [row['k'] - row['k_p']]

    # Cuantities to store
    libration_angles = {m: np.empty(N) for m in modes}
    a_history, e_history, i_history = np.empty(N), np.empty(N), np.empty(N)

    # Add the resonant candidate
    sim.add(m=0.0,
            a=row["a"],
            e=row["e_x"],
            inc=np.deg2rad(row["i_x"]),
            Omega=np.deg2rad(row["om"]),
            omega=np.deg2rad(row["w"]),
            M=np.deg2rad(row["ma"]),
            primary= sim.particles["Sun"],
            name="candidate")

    # Start the simulation and integrate for 5 resonant periods
    resonant_period= row['per_lib']
    times = np.linspace(0, res_cycles*resonant_period, N)

    for j, t in enumerate(times):
        sim.integrate(t)

        # Determinate the orbit of the candidate and the planet
        orbit = sim.particles["candidate"].orbit(primary=sim.particles["Sun"])
        orbit_p = sim.particles[row['planet']].orbit(primary=sim.particles["Sun"])

        # Calculate the resonant angle for each mode
        mean_long = angles.mean_long(orbit.M, orbit.Omega, orbit.omega)
        mean_long_p = angles.mean_long(orbit_p.M, orbit_p.Omega, orbit_p.omega)
        for m in modes:
            libration_angles[m][j] = angles.resonant_angle(row['k_p'], row['k'], mean_long, mean_long_p, orbit.omega, orbit.Omega, m)

        # Store the orbital elements
        a_history[j], e_history[j], i_history[j] = orbit.a, orbit.e, np.rad2deg(orbit.inc)

    libration_angles_unwrapped = np.vstack([np.unwrap(libration_angles[m]) for m in modes])

    return {
        "time": times,
        "modes": modes,
        "sigma": libration_angles_unwrapped,
        "a": a_history,
        "e": e_history,
        "i": i_history,
    }
