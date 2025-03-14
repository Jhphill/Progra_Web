
"""
#EJERCICIO 1
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
# Función ANOVA con pruebas de normalidad y homogeneidad de varianzas
def anova_analysis(*groups, alpha=0.05):
    k = len(groups)  # Número de grupos
    n_total = sum(len(group) for group in groups)  # Total de observaciones
    grand_mean = np.mean([x for group in groups for x in group])  # Media general
    
    # Suma de cuadrados entre grupos (SSB)
    ssb = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)
    dfb = k - 1  # Grados de libertad entre grupos
    msb = ssb / dfb  # Media cuadrática entre grupos
    
    # Suma de cuadrados dentro de los grupos (SSW)
    ssw = sum(sum((x - np.mean(group)) ** 2 for x in group) for group in groups)
    dfw = n_total - k  # Grados de libertad dentro de los grupos
    msw = ssw / dfw  # Media cuadrática dentro de los grupos
    
    # Estadístico F
    F = msb / msw
    p_value = 1 - stats.f.cdf(F, dfb, dfw)  # Valor p
    
    # Prueba de normalidad (Shapiro-Wilk)
    normality_tests = {f"Grupo {i+1}": stats.shapiro(group).pvalue for i, group in enumerate(groups)}
    normality_pass = all(p > alpha for p in normality_tests.values())
    # Prueba de homogeneidad de varianzas (Levene)
    levene_p = stats.levene(*groups).pvalue
    homogeneity_pass = levene_p > alpha 
    # Decisión sobre la hipótesis nula
    reject_h0 = p_value < alpha
    decision = "Se rechaza la hipótesis nula" if reject_h0 else "No se rechaza la hipótesis nula"
    # Resultados
    results = {
        "Grados de libertad (entre grupos)": dfb,
        "Suma de cuadrados (entre grupos)": ssb,
        "Media cuadrática (entre grupos)": msb,
        "Grados de libertad (dentro de grupos)": dfw,
        "Suma de cuadrados (dentro de grupos)": ssw,
        "Media cuadrática (dentro de grupos)": msw,
        "Valor F": F,
        "Valor P": p_value,
        "Normalidad (Shapiro-Wilk, p-valores)": normality_tests,
        "Homogeneidad de varianzas (Levene, p-valor)": levene_p,
        "Decisión": decision
    }
    # Gráfico de caja para visualización
    plt.boxplot(groups, tick_labels=[f"Método {i+1}" for i in range(len(groups))])
    plt.title("Boxplot de los tiempos por método")
    plt.ylabel("Tiempo (min)")
    plt.show()  
    return results
# Datos de los métodos de entrenamiento
metodo1 = [15,16,14,15,17]
metodo2 = [14,13,15,16,14]
metodo3 = [13,12,11,14,11]
# Ejecutar ANOVA con los datos
anova_results = anova_analysis(metodo1, metodo2, metodo3)
anova_results
"""
# ejercicio 2
"""
import numpy as np
import scipy.stats as stats

data = {
    "Control": [50, 65, 72, 46, 38, 29, 70, 85, 72, 40, 57, 59],
    "25mg": [49, 47, 30, 602, 62, 60, 19, 28, 56, 62, 44, 40],
    "50mg": [20, 59, 64, 61, 28, 47, 29, 41, 60, 57, 61, 38],
    "100mg": [20, 23, 38, 31, 27, 16, 27, 18, 22, 12, 24, 11],
    "125mg": [18, 30, 22, 26, 31, 11, 15, 12, 31, 36, 16, 13]
}

# Extraer los valores en listas para realizar ANOVA
groups = list(data.values())

# Aplicar ANOVA
F, p_value = stats.f_oneway(*groups)

# Grados de libertad
k = len(groups)  # Número de grupos
n = sum(len(g) for g in groups)  # Número total de observaciones
df_between = k - 1
df_within = n - k

# Calcular suma de cuadrados
grand_mean = np.mean([item for sublist in groups for item in sublist])
SSB = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
SSW = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)

# Medias cuadráticas
MSB = SSB / df_between
MSW = SSW / df_within

# Mostrar resultados
print(f"Grados de libertad entre grupos: {df_between}")
print(f"Grados de libertad dentro de grupos: {df_within}")
print(f"Suma de cuadrados entre grupos: {SSB:.2f}")
print(f"Suma de cuadrados dentro de grupos: {SSW:.2f}")
print(f"Media cuadrática entre grupos: {MSB:.2f}")
print(f"Media cuadrática dentro de grupos: {MSW:.2f}")
print(f"Valor F: {F:.2f}")
print(f"Valor P: {p_value:.5f}")

# Evaluar hipótesis
alpha = 0.05
if p_value < alpha:
    print("Rechazamos la hipótesis nula. Al menos una dosis tiene un efecto significativo.")
else:
    print("No se puede rechazar la hipótesis nula. El medicamento no tiene un efecto significativo.")
"""



# Ejercicio 3

"""
import numpy as np
import scipy.stats as stats

# Datos de emisión de CO2 por tipo de máquina
machine_data = {
    "I": [24, 26, 29],
    "II": [27, 30, 32],
    "III": [26, 27, 30],
    "IV": [25, 28, 28],
    "V": [28, 29, 31]
}

# Extraer los valores en listas para realizar ANOVA
groups = list(machine_data.values())

# Aplicar ANOVA
F, p_value = stats.f_oneway(*groups)

# Grados de libertad
k = len(groups)  # Número de grupos
n = sum(len(g) for g in groups)  # Número total de observaciones
df_between = k - 1
df_within = n - k

# Calcular suma de cuadrados
grand_mean = np.mean([item for sublist in groups for item in sublist])
SSB = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
SSW = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)

# Medias cuadráticas
MSB = SSB / df_between
MSW = SSW / df_within

# Mostrar resultados
print(f"Grados de libertad entre grupos: {df_between}")
print(f"Grados de libertad dentro de grupos: {df_within}")
print(f"Suma de cuadrados entre grupos: {SSB:.2f}")
print(f"Suma de cuadrados dentro de grupos: {SSW:.2f}")
print(f"Media cuadrática entre grupos: {MSB:.2f}")
print(f"Media cuadrática dentro de grupos: {MSW:.2f}")
print(f"Valor F: {F:.2f}")
print(f"Valor P: {p_value:.5f}")

# Evaluar hipótesis
alpha = 0.05
if p_value < alpha:
    print("Rechazamos la hipótesis nula. Existen diferencias significativas entre los tipos de máquinas.")
else:
    print("No se puede rechazar la hipótesis nula. No hay diferencias significativas entre los tipos de máquinas.")
"""

# Ejercicio 4 
import numpy as np
import pandas as pd
import scipy.stats as stats

# Datos del porcentaje de lípidos por marca y analista
data = {
    "Marca": ["I"] * 4 + ["II"] * 4 + ["III"] * 4,
    "Analista": ["A", "B", "C", "D"] * 3,
    "Lípidos": [8.16, 8.67, 7.91, 8.93,  # Marca I
                10.20, 9.18, 8.41, 7.39, # Marca II
                9.44, 7.65, 7.14, 8.41]  # Marca III
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Aplicar ANOVA de dos factores
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Modelo ANOVA
modelo = ols('Lípidos ~ C(Marca) + C(Analista)', data=df).fit()
anova_tabla = sm.stats.anova_lm(modelo, typ=2)

# Mostrar resultados
print(anova_tabla)

# Evaluar hipótesis
alpha = 0.05
if anova_tabla["PR(>F)"]["C(Marca)"] < alpha:
    print("Hay diferencias significativas entre las marcas de galletas.")
else:
    print("No hay diferencias significativas entre las marcas de galletas.")

if anova_tabla["PR(>F)"]["C(Analista)"] < alpha:
    print("Hay diferencias significativas entre los analistas.")
else:
    print("No hay diferencias significativas entre los analistas.")
