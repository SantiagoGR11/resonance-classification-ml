# Importación de librerías
import numpy as np
import pandas as pd

# Criterio de selección para posible resonante
def posible_resonant(a, sigma_a, a_res, Delta_R, GM = 4*np.pi**2): # G·M_sun = GM expressed in AU^3/year^2
    
    # Ancho de las fluctuaciones en a al resonar
    n_res = np.sqrt(GM / a_res**3)
    Delta_a_res = np.sqrt(8/3)/n_res * np.sqrt(Delta_R)

    return np.abs(a - a_res) <= (sigma_a + Delta_a_res)


def build_resonant_candidates(catalog, strengths):

    # Mallado de la excentricidad y la inclinación
    excentricities = np.linspace(0, 0.9, 10)
    inclinations = np.linspace(0, 180, 10)

    # Ordenado por las resonancias más fuertes para cada configuración
    SR_ranking = pd.DataFrame(columns=strengths.columns)

    for inc in inclinations:
        for exc in excentricities:
            config_filter = strengths[(np.isclose(strengths['i'], inc)) & (np.isclose(strengths['e'], exc))]
            res_ei_ranking = config_filter.sort_values(by='SR', ascending=False).head(5)
            SR_ranking = pd.concat([SR_ranking, res_ei_ranking.head(5)], ignore_index = True)

    # Corte por si entró alguna resonancia despreciable
    SR_ranking = SR_ranking[SR_ranking["SR"]>1e-8].convert_dtypes()

    # Solo los candidatos con la órbita bien definida
    catalog = catalog[catalog["orbit_status"]=="ok"]


    # Se busca a lo largo del catálogo y por las regiones de (e, i) ya definidas
    de = excentricities[1] - excentricities[0]
    di = inclinations[1] - inclinations[0]

    # Adición de índices a las celdas de los cuerpos
    catalog["e_idx"] = np.clip(np.rint(catalog["e"] / de).astype(int), 0, len(excentricities)-1)
    catalog["i_idx"] = np.clip(np.rint(catalog["i"] / di).astype(int), 0, len(inclinations)-1)

    # Adición de índices a las celdas de las resonancias
    SR_ranking["e_idx"] = np.clip(np.rint(SR_ranking["e"] / de).astype(int), 0, len(excentricities)-1)
    SR_ranking["i_idx"] = np.clip(np.rint(SR_ranking["i"] / di).astype(int), 0, len(inclinations)-1)

    # Todas las combinaciones cuerpo-resonancia físicamente relevantes
    configs_res_body = catalog.merge(SR_ranking[["e_idx", "i_idx", "e", "i", "resonance", "planet", "a_res", "SR", "Delta_R", "per_lib", "d_min/R_H", "close_encounter_fraction", "status"]], on=["e_idx", "i_idx"], how="inner")

    # Prefiltro dinámico, completamente vectorizado
    mask = posible_resonant(
        configs_res_body["a"].to_numpy(),
        configs_res_body["sigma_a"].to_numpy(),
        configs_res_body["a_res"].fillna(0.0).to_numpy(),
        configs_res_body["Delta_R"].to_numpy())

    candidates = configs_res_body.loc[mask].copy()
    candidates[["k_p", "k"]] = (candidates["resonance"].str.split(":", expand=True).astype(int))

    return candidates