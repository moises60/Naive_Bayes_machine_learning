import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilos de gráficos
sns.set(style="whitegrid")

# Importar el dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Análisis Exploratorio de Datos (EDA)
# Mostrar las primeras filas
print("Primeras 5 filas del dataset:")
print(dataset.head())

# Descripción estadística
print("\nDescripción estadística del dataset:")
print(dataset.describe())

# Verificar balance de clases
print("\nBalance de clases:")
print(dataset['Comprará'].value_counts())

# Visualizar la distribución de las variables
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(dataset['Edad'], kde=True, bins=20)
plt.title('Distribución de la Edad')

plt.subplot(1, 2, 2)
sns.histplot(dataset['Sueldo Estimado'], kde=True, bins=20)
plt.title('Distribución del Sueldo Estimado')
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y='Edad', data=dataset)
plt.title('Boxplot de Edad')

plt.subplot(1, 2, 2)
sns.boxplot(y='Sueldo Estimado', data=dataset)
plt.title('Boxplot de Sueldo Estimado')
plt.tight_layout()
plt.show()

# Scatter plot con ajuste de regresión
sns.lmplot(x='Edad', y='Sueldo Estimado', hue='Comprará', data=dataset, 
           markers=['o', 'x'], palette='Set1', fit_reg=False, height=6, aspect=1.2)
plt.title('Edad vs Sueldo Estimado')
plt.show()

# Matriz de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

# Preparar los datos para el modelo
X = dataset[['Edad', 'Sueldo Estimado']].values
y = dataset['Comprará'].values

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# Escalado de características
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear una función para entrenar y evaluar modelos
def entrenar_evaluar_modelo(clf, nombre_modelo):
    # Entrenar el modelo
    clf.fit(X_train, y_train)
    
    # Predecir en el conjunto de prueba
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Evaluar el modelo
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score

    print(f"\n--- {nombre_modelo} ---")
    print("Matriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy * 100:.2f}%")

    # Curva ROC
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {nombre_modelo}')
    plt.legend(loc="lower right")
    plt.show()

    # Matriz de Confusión Gráfica
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap='Blues')
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.show()

    # Frontera de decisión
    plot_decision_boundary(clf, X_test, y_test, nombre_modelo)

# Función para graficar la frontera de decisión
def plot_decision_boundary(clf, X, y, nombre_modelo):
    from matplotlib.colors import ListedColormap
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(
        np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01),
        np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    )
    plt.figure(figsize=(8,6))
    plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.5, cmap=ListedColormap(('red', 'green')))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(('red', 'green')), edgecolor='k')
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    plt.title(f'Frontera de Decisión - {nombre_modelo}')
    plt.xlabel('Edad (Estandarizada)')
    plt.ylabel('Sueldo Estimado (Estandarizado)')
    plt.show()

#Si se importan los clasificadores adecuados se puede crear una lista de modelos, aunque en esta ocasión solo utilizo naive. 
from sklearn.naive_bayes import GaussianNB


# Modeloo
modelos = [
    (GaussianNB(), 'Naive Bayes'),

]

# Entrenar y evaluar modelo
for clf, nombre_modelo in modelos:
    entrenar_evaluar_modelo(clf, nombre_modelo)

