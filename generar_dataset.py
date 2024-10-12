# generar_dataset.py

import numpy as np
import pandas as pd

# Establecer una semilla para reproducibilidad
np.random.seed(0)

# Generar datos
num_muestras = 300

# Generar edades entre 18 y 60
edades = np.random.randint(18, 61, size=num_muestras)

# Generar sueldos estimados entre 20,000 y 150,000
sueldos = np.random.randint(20000, 150001, size=num_muestras)

# Generar una variable objetivo (0: No Compra, 1: Compra) basada en alguna lógica
# Por ejemplo, mayores ingresos y edades pueden tener mayor probabilidad de compra
probabilidad_compra = (sueldos / 170000) + (edades / 600)
probabilidad_compra = probabilidad_compra / probabilidad_compra.max()  # Normalizar entre 0 y 1
compras = np.random.binomial(1, probabilidad_compra)

# Crear un DataFrame
dataset = pd.DataFrame({
    'Edad': edades,
    'Sueldo Estimado': sueldos,
    'Comprará': compras
})

# Guardar el dataset en un archivo CSV
dataset.to_csv('Social_Network_Ads.csv', index=False)

print("Dataset generado y guardado como 'Social_Network_Ads.csv'.")
