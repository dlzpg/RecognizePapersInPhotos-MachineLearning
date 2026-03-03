import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys 

def deteccionEsquinas(direccionImagen):
    foto = cv2.imread(direccionImagen, cv2.IMREAD_COLOR)
    fotoRGB = cv2.cvtColor(foto, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.GaussianBlur(gray,(5,5),0)

    ret,outs_img = cv2.threshold(filtered_image,127,255,cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    eroded_image = cv2.erode(outs_img, kernel)

    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    opened_image = cv2.morphologyEx(eroded_image, cv2.MORPH_OPEN, opening_kernel)
    dilated_image = cv2.dilate(opened_image, kernel)
    imgR = cv2.resize(dilated_image,(dilated_image.shape[1]//10,dilated_image.shape[0]//10))
    
    blockSize = 10
    ksize = 3 
    k = 0.1 
    esquinosidad = cv2.cornerHarris(imgR,blockSize,ksize,k)
    if len(imgR.shape) == 2:
        imgO = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

    esquinosidad_dilatada = cv2.dilate(esquinosidad, None)

    # Aplicar un umbral para seleccionar puntos de interés
    threshold = 0.1 * esquinosidad_dilatada.max() 
    imgO[esquinosidad_dilatada > threshold] = [0, 0, 255]  
    sift = cv2.ORB_create(edgeThreshold=10)
    keypoints, descriptors = sift.detectAndCompute(imgO, None)
    img_sift = cv2.drawKeypoints(imgO, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    des_test = descriptors  
    kp_test = keypoints    

    # Aplicar BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors, des_test)
    matches = sorted(matches, key=lambda x: x.distance)

    # Definir los límites de la imagen en base a los keypoints
    x_vals = [kp.pt[0] for kp in kp_test]
    y_vals = [kp.pt[1] for kp in kp_test]
    h, w = imgO.shape[:2]

    quadrants = get_quadrants(imgO)

    # Seleccionar el mejor punto por cuadrante
    best_points = select_best_points_by_quadrant(kp_test, matches, quadrants, imgO)

    dstCua = np.array([[0, 0], [2480, 0], [2480, 3508], [0, 3508]], np.float32)
    # Factor de escala usado para redimensionar
    scale_x = dilated_image.shape[1] / imgR.shape[1]
    scale_y = dilated_image.shape[0] / imgR.shape[0]

    # Escalar los puntos detectados al tamaño original
    srcCua = np.array([
        (best_points['top_left'][0] * scale_x, best_points['top_left'][1] * scale_y),
        (best_points['top_right'][0] * scale_x, best_points['top_right'][1] * scale_y),
        (best_points['bottom_right'][0] * scale_x, best_points['bottom_right'][1] * scale_y),
        (best_points['bottom_left'][0] * scale_x, best_points['bottom_left'][1] * scale_y)
    ], dtype=np.float32)

    persp_mat = cv2.getPerspectiveTransform(srcCua,dstCua)
    foto2 = cv2.warpPerspective(fotoRGB,persp_mat,(2480,3508))

    return foto2
    
    #plt.title(direccionImagen)
    #plt.subplot(1,2,1)
    #plt.imshow(fotoRGB)
    #plt.subplot(1,2,2)
    #plt.imshow(foto2)
    #plt.show()

def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

#Funcion para definir los cuadrantes
def get_quadrants(imgO):
    h, w = imgO.shape[:2]
    quadrants = {
        'top_left': (0, h // 2, 0, w // 2),
        'top_right': (0, h // 2, w // 2, w),
        'bottom_left': (h // 2, h, 0, w // 2),
        'bottom_right': (h // 2, h, w // 2, w),
    }
    return quadrants

# Funcion para seleccionar las esquinas priorizando puntos cercanos a las esquinas originales
def select_best_points_by_quadrant(kp_test, matches, quadrants, imgO):
    best_points = {}

    #Esquinas de la imagen
    image_corners = [(0, 0), (0, imgO.shape[1]), (imgO.shape[0], 0), (imgO.shape[0], imgO.shape[1])]

    #Constante para evitar división por cero
    epsilon = 1e-6

    for q, (y1, y2, x1, x2) in quadrants.items():
        best_match = None
        min_distance = float('inf')
        min_corner_distance = float('inf')
        best_point = None
        max_priority = -float('inf')

        corners = {
            'top_left': (x1, y1),
            'top_right': (x2, y1),
            'bottom_left': (x1, y2),
            'bottom_right': (x2, y2),
        }
        corner = corners[q]

        for match in matches:
            x, y = kp_test[match.trainIdx].pt
            if x1 <= x < x2 and y1 <= y < y2:
                # Calcular la distancia al descriptor de referencia
                if match.distance < min_distance:
                    min_distance = match.distance
                    best_match = match
                    best_point = kp_test[best_match.trainIdx].pt
                    
                # Calcular la distancia a la esquina del cuadrante
                dist_to_corner = euclidean_distance((x, y), corner)

                # Calcular la prioridad basada en la proximidad a la esquina
                corner_priority = 1 / (dist_to_corner + epsilon) 

                # Si el punto es más cercano a la esquina, darle más prioridad
                if corner_priority > max_priority:
                    max_priority = corner_priority
                    best_point = (x, y)
                    best_match = match

        if best_point:
            best_points[q] = best_point

    return best_points



#Uso de sys para la obtención de la dirección de la imagen
if __name__ == "__main__":
    direccion = sys.argv[1]
    deteccionEsquinas(direccion)