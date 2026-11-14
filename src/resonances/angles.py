import numpy as np

# Longitud del perihelio
def peri_long(Omega, omega):
    return Omega+omega

# Longitud media
def mean_long(M, Omega, omega):
    return M+Omega+omega

# Ángulo resonante
def resonant_angle(k_p, k, lambda_body, lambda_planet, omega, Omega, m):
    sigma = k*lambda_body - k_p*lambda_planet - (k-k_p)*Omega - m*omega
    return np.mod(sigma, 2*np.pi)

# Modos resonantes
def resonant_modes(k_p, k):
    mmax = k_p+k
    modes = np.arange(-mmax, mmax+2, 2)
    return modes