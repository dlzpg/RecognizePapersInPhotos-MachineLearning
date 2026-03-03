import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition, metrics
from scanner import deteccionEsquinas

# ---------------------- CONFIGURACIÓN ----------------------
BASE_DIR = "./PracticaObligatoria2_2025/MUESTRA/"
APRENDIZAJE_DIR = BASE_DIR + "Aprendizaje/"
TEST_DIR = BASE_DIR + "Test/"
IMAGE_SIZE = (400, 300)

clases_dic = {'Comics': 0, 'Libros': 1, 'Manuscrito': 2, 'Mecanografiado': 3, 'Tickets': 4}

# ---------------------- FUNCIONES ----------------------
def cargar_imagenes_y_etiquetas(directorio):
    X, y = [], []
    for clase, etiqueta in clases_dic.items():
        carpeta = os.path.join(directorio, clase)
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                ruta = os.path.join(carpeta, archivo)
                img = cv2.imread(ruta)
                if img is not None:
                    img = cv2.resize(img, IMAGE_SIZE)
                    X.append(img.flatten())
                    y.append(etiqueta)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def cargar_imagenes_y_etiquetas_c3(directorio):
    X, y = [], []
    for clase, etiqueta in clases_dic.items():
        carpeta = os.path.join(directorio, clase)
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                ruta = os.path.join(carpeta, archivo)
                img = deteccionEsquinas(ruta)
                if img is not None:
                    img = cv2.resize(img, IMAGE_SIZE)
                    X.append(img.flatten())
                    y.append(etiqueta)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# ---------------------- C1 ----------------------
X_train_c1, y_train_c1 = cargar_imagenes_y_etiquetas(APRENDIZAJE_DIR)
X_test_c1, y_test_c1 = cargar_imagenes_y_etiquetas(TEST_DIR)

svm_c1 = SVC(kernel='linear')
svm_c1.fit(X_train_c1, y_train_c1)
y_pred_c1 = svm_c1.predict(X_test_c1)

acc_c1 = metrics.accuracy_score(y_test_c1, y_pred_c1)
prec_c1 = metrics.precision_score(y_test_c1, y_pred_c1, average='weighted')
rec_c1 = metrics.recall_score(y_test_c1, y_pred_c1, average='weighted')
print(f"Accuracy del clasificador C1 (SVM): {acc_c1:.2%}")
print(f"Precision: {prec_c1:.2%}")
print(f"Recall: {rec_c1:.2%}")

cm_c1 = confusion_matrix(y_test_c1, y_pred_c1)
ConfusionMatrixDisplay(confusion_matrix=cm_c1, display_labels=list(clases_dic.keys())).plot(cmap=plt.cm.Blues)
plt.title("C1 - SVM (RGB)")
plt.show()

# ---------------------- C2 ----------------------
# PCA
pca_c2 = decomposition.PCA(n_components=30)
X_train_pca_c2 = pca_c2.fit_transform(X_train_c1)
X_test_pca_c2 = pca_c2.transform(X_test_c1)

# LDA
lda_c2 = LinearDiscriminantAnalysis(n_components=4)
X_train_lda_c2 = lda_c2.fit_transform(X_train_pca_c2, y_train_c1)
X_test_lda_c2 = lda_c2.transform(X_test_pca_c2)

# SVM
svm_c2 = SVC(kernel='linear')
svm_c2.fit(X_train_lda_c2, y_train_c1)
y_pred_c2 = svm_c2.predict(X_test_lda_c2)

acc_c2 = metrics.accuracy_score(y_test_c1, y_pred_c2)
prec_c2 = metrics.precision_score(y_test_c1, y_pred_c2, average='weighted')
rec_c2 = metrics.recall_score(y_test_c1, y_pred_c2, average='weighted')
print(f"\nC2 Accuracy (PCA+LDA+SVM): {acc_c2:.2%}")
print(f"Precision: {prec_c2:.2%}")
print(f"Recall: {rec_c2:.2%}")

cm_c2 = confusion_matrix(y_test_c1, y_pred_c2)
ConfusionMatrixDisplay(confusion_matrix=cm_c2, display_labels=list(clases_dic.keys())).plot(cmap=plt.cm.Oranges)
plt.title("C2 - PCA + LDA + SVM (RGB)")
plt.show()

# ---------------------- C3 ----------------------
X_train_c3, y_train_c3 = cargar_imagenes_y_etiquetas_c3(APRENDIZAJE_DIR)
X_test_c3, y_test_c3 = cargar_imagenes_y_etiquetas_c3(TEST_DIR)

svm_c3 = SVC(kernel='linear')
svm_c3.fit(X_train_c3, y_train_c3)
y_pred_c3 = svm_c3.predict(X_test_c3)


acc_c3 = metrics.accuracy_score(y_test_c3, y_pred_c3)
prec_c3 = metrics.precision_score(y_test_c3, y_pred_c3, average='weighted')
rec_c3 = metrics.recall_score(y_test_c3, y_pred_c3, average='weighted')
print(f"\nC3 Accuracy (SVM sobre rectificadas): {acc_c3:.2%}")
print(f"Precision: {prec_c3:.2%}")
print(f"Recall: {rec_c3:.2%}")

cm_c3 = confusion_matrix(y_test_c3, y_pred_c3)
ConfusionMatrixDisplay(confusion_matrix=cm_c3, display_labels=list(clases_dic.keys())).plot(cmap=plt.cm.Purples)
plt.title("C3 - SVM (rectificadas)")
plt.show()

# ---------------------- C4 ----------------------
# PCA
pca_c4 = decomposition.PCA(n_components=30)
X_train_pca_c4 = pca_c4.fit_transform(X_train_c3)
X_test_pca_c4 = pca_c4.transform(X_test_c3)

# LDA
lda_c4 = LinearDiscriminantAnalysis(n_components=2)
X_train_lda_c4 = lda_c4.fit_transform(X_train_pca_c4, y_train_c3)
X_test_lda_c4 = lda_c4.transform(X_test_pca_c4)

# SVM
svm_c4 = SVC(kernel='linear')
svm_c4.fit(X_train_lda_c4, y_train_c3)
y_pred_c4 = svm_c4.predict(X_test_lda_c4)


acc_c4 = metrics.accuracy_score(y_test_c3, y_pred_c4)
prec_c4 = metrics.precision_score(y_test_c3, y_pred_c4, average='weighted')
rec_c4 = metrics.recall_score(y_test_c3, y_pred_c4, average='weighted')
print(f"\nC4 Accuracy (PCA + LDA + SVM, rectificadas): {acc_c4:.2%}")
print(f"Precision: {prec_c4:.2%}")
print(f"Recall: {rec_c4:.2%}")

cm_c4 = confusion_matrix(y_test_c3, y_pred_c4)
ConfusionMatrixDisplay(confusion_matrix=cm_c4, display_labels=list(clases_dic.keys())).plot(cmap=plt.cm.Greens)
plt.title("C4 - PCA + LDA + SVM (rectificadas)")
plt.show()



#-----------------------3.5---------------------
print(f"{'Clasificador':<15}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}")
print("-" * 63)
print(f"{'C1':<15}{acc_c1*100:<12.2f}{prec_c1*100:<12.2f}{rec_c1*100:<12.2f}")
print(f"{'C2':<15}{acc_c2*100:<12.2f}{prec_c2*100:<12.2f}{rec_c2*100:<12.2f}")
print(f"{'C3':<15}{acc_c3*100:<12.2f}{prec_c3*100:<12.2f}{rec_c3*100:<12.2f}")
print(f"{'C4':<15}{acc_c4*100:<12.2f}{prec_c4*100:<12.2f}{rec_c4*100:<12.2f}")

#-----------------------Comprobar entrada---------------------

foto = sys.argv[1]

# --- Para C1 y C2 ---
img = cv2.imread(foto)
img_rgb = cv2.resize(img, IMAGE_SIZE)
img_flat_rgb = img_rgb.flatten().reshape(1, -1).astype(np.float32)

# C1
pred_c1 = svm_c1.predict(img_flat_rgb)[0]
print(f"\nPredicción C1: {list(clases_dic.keys())[pred_c1]}")

# C2
img_pca = pca_c2.transform(img_flat_rgb)
img_lda = lda_c2.transform(img_pca)
pred_c2 = svm_c2.predict(img_lda)[0]
print(f"Predicción C2: {list(clases_dic.keys())[pred_c2]}")

# --- Para C3 y C4 ---
img_rect = deteccionEsquinas(foto)
img_rect = cv2.resize(img_rect, IMAGE_SIZE)
img_flat_rect = img_rect.flatten().reshape(1, -1).astype(np.float32)

# C3
pred_c3 = svm_c3.predict(img_flat_rect)[0]
print(f"Predicción C3: {list(clases_dic.keys())[pred_c3]}")

# C4
img_pca_r = pca_c4.transform(img_flat_rect)
img_lda_r = lda_c4.transform(img_pca_r)
pred_c4 = svm_c4.predict(img_lda_r)[0]
print(f"Predicción C4: {list(clases_dic.keys())[pred_c4]}")