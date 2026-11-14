# Importación de librerías
import numpy as np
from math import gcd


# Generación de resonancias del tipo 1:N (hasta N=20)
N_max = 20
resonances = {(1, N) for N in range(1, N_max + 1)}


# Resonancias generales para 2 cuerpos
PQ_max = 8
for k_p in range(1, PQ_max + 1):
    for k in range(1, PQ_max + 1):
        g = gcd(k_p, k)
        resonances.add((k_p // g, k // g))

resonances = sorted(resonances)


# Parámetros de planetas gigantes
a_J, r_J, m_J = 5.20, 6.991e7, 9.547e-4 # AU, m, masas solares
a_S, r_S, m_S = 9.58, 5.823e7, 2.858e-4 # AU, m, masas solares
a_U, r_U, m_U = 19.19, 2.536e7, 4.366e-5 # AU, m, masas solares
a_N, r_N, m_N = 30.07, 2.462e7, 5.151e-5 # AU, m, masas solares

planets = ["Jupiter", "Saturn", "Uranus", "Neptune"]
a_planet = [a_J, a_S, a_U, a_N]
m_planet = [m_J, m_S, m_U, m_N]


# Mallado de la excentricidad y la inclinación
excentricities = np.linspace(0, 0.9, 10)
inclinations = np.deg2rad(np.linspace(0, 180, 10))


# Validación de la resonancia k_p:k
def a_resonance_validation(k, k_p, a_p, m_p, a_min=a_J-0.05, a_max=a_N+0.05):
    a_res = a_p/(1 + m_p)**(1/3) * (k/k_p)**(2/3)
    if a_min <= a_res <= a_max:
        return a_res, True
    else:
        return a_res, False


# Posición del planeta
def planet_position(lambda_p, a_p):
    x_p, y_p, z_p = a_p*np.cos(lambda_p), a_p*np.sin(lambda_p), np.zeros_like(lambda_p)
    r_p = np.array([x_p, y_p, z_p])
    return r_p


# Posición del cuerpo
def body_position(sigma, lambda_p, a, e, i, k, k_p, omega=np.deg2rad(60)):
    M = (sigma + k_p*lambda_p - k_p*omega) / k
    M = np.mod(np.asarray(M, dtype=float), 2*np.pi)

    ## Determinación de la anomalía excéntrica, E
    E, n_max, tol=np.full_like(M, np.pi), 100, 1e-12
    for n in range(n_max):
        dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
        E -= dE

        if np.max(np.abs(dE)) < tol:
            break

    ## Posición orbital
    x_orb, y_orb, z_orb = a*(np.cos(E)-e), a*np.sqrt(1-e**2)*np.sin(E), np.zeros_like(lambda_p)

    ## Rotaciones según i y omega
    x = np.cos(omega)*x_orb - np.sin(omega)*y_orb
    y = (np.sin(omega)*x_orb + np.cos(omega)*y_orb) * np.cos(i)
    z = (np.sin(omega)*x_orb + np.cos(omega)*y_orb) * np.sin(i)

    ## Posición del cuerpo
    r = np.array([x, y, z])
    return r


# Cuantización del efecto de la resonancia
def disturbing_integrand(r, r_p, a_p, m_p, G=4*np.pi**2): #G expresada en AU^3 · M_sun^-1 · years^-2
    distance = np.linalg.norm(r_p-r, axis=0)
    R = G * m_p * (1/distance - np.sum(r * r_p, axis=0)/a_p**3)
    return R, distance


# Integración de la función
def resonant_disturbing_function(sigma, k, k_p, a, e, i, a_p, m_p,n_orbit=64):
    ## Pasos totales
    n = k * n_orbit

    ## Intervalo de integración
    lambda_p = np.linspace(0, 2*np.pi*k, n+1)
    l_p = lambda_p[None, :]

    ## Integrando en función de l_p
    r_p = planet_position(l_p, a_p)
    r = body_position(sigma, l_p, a, e, i, k, k_p)
    R, distance = disturbing_integrand(r, r_p, a_p, m_p)

    ## Integración sin restricciones
    R_sigma = 1/(2*np.pi*k) * np.trapezoid(R, l_p)

    ## Tratamiento de encuentros cercanos (d<3R_H)
    R_H = a_p * (m_p/3)**(1/3)
    d_RH = distance/R_H
    d_min_sigma = np.min(d_RH, axis=-1)
    close_fraction_sigma = np.mean(d_RH < 3, axis=-1)

    return R_sigma, d_min_sigma, close_fraction_sigma


# Fuerza de la resonancia
def resonance_strength(k, k_p, a, e, i, a_p, m_p, n_sigma=180):

    ## Intervalo de sigma
    sigma_grid = np.linspace(0, 2*np.pi, n_sigma, endpoint=False)
    sigma = sigma_grid[:, None]

    ## Valor de R para cada sigma
    R, d_mins, close_fractions = resonant_disturbing_function(sigma, k, k_p, a, e, i, a_p, m_p)

    ## Distancias seguras (d>3R_H)
    safe = d_mins >= 3
    sigma_safe = sigma_grid[safe]
    R_safe = R[safe]

    ## En caso de no existir valores de sigma seguros
    if R_safe.size == 0:
        return np.nan, np.nan, np.nan, np.min(d_mins), np.mean(close_fractions), sigma_safe, R_safe

    ## Fuerza y otras cantidades relevantes de la resonancia
    SR = np.mean(R_safe) - np.min(R_safe)
    Delta_R = np.max(R_safe) - np.min(R_safe)
    d_min = np.min(d_mins)
    close_fraction = np.mean(close_fractions)
    idx_R_min = np.argmin(R_safe)
    R_ss = (R_safe[(idx_R_min+1) % len(R_safe)]- 2*R_safe[idx_R_min]+ R_safe[(idx_R_min-1) % len(R_safe)]) / (sigma_grid[1]-sigma_grid[0])**2
    if R_ss > 0: 
        per_lib = 2*np.pi*a / (k*np.sqrt(3*R_ss))
    else:
        per_lib = np.nan
    return SR, Delta_R, per_lib, d_min, close_fraction, sigma_safe, R_safe

# Ejecución y almacenamiento de todas las configuraciones posibles
def compute_resonance_strengths():
    results, executed_configs, total_configs = [], 0, len(planets)*len(resonances)*len(inclinations)*len(excentricities)
    for idx_planet, planet in enumerate(planets):
        a_p, m_p = a_planet[idx_planet], m_planet[idx_planet]
        for (k_p, k) in resonances:
            a_res, a_res_bool = a_resonance_validation(k, k_p, a_p, m_p)
            if a_res_bool == False:
                results.append({
                    "resonance": f"{k_p}:{k}",
                    "planet": planet,
                    "i": np.nan,
                    "e": np.nan,
                    "a_res": a_res,
                    "SR": np.nan,
                    "Delta_R": np.nan,
                    "per_lib": np.nan,
                    "d_min/R_H": np.nan,
                    "close_encounter_fraction": np.nan,
                    "status": "OutOfRange"
                })

                executed_configs += len(inclinations)*len(excentricities)
                print(f"\rExecuted configurations: {executed_configs}/{total_configs}",end="", flush=True)
                continue

            for i in inclinations:
                for e in excentricities:
                    SR, Delta_R, per_lib, d_min, close_fraction, sigma_grid, R = resonance_strength(k, k_p, a_res, e, i, a_p, m_p)
                    results.append({
                                    "resonance": f"{k_p}:{k}",
                                    "planet": planet,
                                    "i": np.rad2deg(i),
                                    "e": e,
                                    "a_res": a_res,
                                    "SR": SR,
                                    "Delta_R": Delta_R,
                                    "per_lib": per_lib,
                                    "d_min/R_H": d_min,
                                    "close_encounter_fraction": close_fraction,
                                    "status": "Valid"
                                })

                    executed_configs += 1
                    print(f"\rExecuted configurations: {executed_configs}/{total_configs}",end="", flush=True)

    return results
    