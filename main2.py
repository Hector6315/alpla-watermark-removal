import os
import cv2
import numpy as np
from src import *

# Carpeta de entrada
folder = "./images/fotolia"

def remove_watermark_from_folder(folder):
    # Asegúrate de que todas las imágenes tengan el mismo tamaño
    gx, gy, gxlist, gylist = estimate_watermark(folder)
    cropped_gx, cropped_gy = crop_watermark(gx, gy)
    W_m = poisson_reconstruct(cropped_gx, cropped_gy)

    # Obtén la lista de imágenes en la carpeta
    image_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    for image_file in image_files:
        # Construye la ruta completa de la imagen
        image_path = os.path.join(folder, image_file)
        
        # Lee la imagen
        img = cv2.imread(image_path)
        
        # Detecta la marca de agua en la imagen
        im, start, end = watermark_detector(img, cropped_gx, cropped_gy)

        # Estimación de la transparencia alpha
        num_images = len(gxlist)
        J, img_paths = get_cropped_images(folder, num_images, start, end, cropped_gx.shape)
        Wm = W_m - W_m.min()
        alph_est = estimate_normalized_alpha(J, Wm, len(J))
        alph = np.stack([alph_est, alph_est, alph_est], axis=2)
        C, est_Ik = estimate_blend_factor(J, Wm, alph)

        alpha = alph.copy()
        for i in range(3):
            alpha[:,:,i] = C[i]*alpha[:,:,i]

        Wm = Wm + alpha*est_Ik

        W = Wm.copy()
        for i in range(3):
            W[:,:,i]/=C[i]

        # Resuelve para todas las imágenes
        Wk, Ik, W, alpha1 = solve_images(J, W_m, alpha, W)

        # Guarda la imagen procesada
        output_image_path = os.path.join(folder, "processed_" + image_file)
        cv2.imwrite(output_image_path, Ik[0])

# Llama a la función para eliminar la marca de agua de todas las imágenes en la carpeta
remove_watermark_from_folder(folder)

print("Proceso completado.")
