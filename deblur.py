import cv2
import numpy as np
import os

def kernel_psf(angle, d, size=3):
    kernel = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    size2 = size // 2
    A[:,2] = (size2, size2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kernel = cv2.warpAffine(kernel, A, (size, size), flags=cv2.INTER_CUBIC)
    return kernel

def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    copy_img = np.copy(img)
    copy_img = np.fft.fft2(copy_img)
    kernel = np.fft.fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    copy_img = copy_img * kernel
    copy_img = np.abs(np.fft.ifft2(copy_img))
    return copy_img

def deblur_image(image_path):
    a = 2.2
    ang = np.deg2rad(90)
    d = 20

    ip_image = cv2.imread(image_path)

    b, g, r = cv2.split(ip_image)

    img_b = np.float32(b) / 255.0
    img_g = np.float32(g) / 255.0
    img_r = np.float32(r) / 255.0

    psf = kernel_psf(ang, d)

    filtered_img_b = wiener_filter(img_b, psf, K=0.0060)
    filtered_img_g = wiener_filter(img_g, psf, K=0.0060)
    filtered_img_r = wiener_filter(img_r, psf, K=0.0060)

    filtered_img = cv2.merge((filtered_img_b, filtered_img_g, filtered_img_r))

    filtered_img = np.clip(filtered_img * 255, 0, 255)
    filtered_img = np.uint8(filtered_img)

    filtered_img = cv2.convertScaleAbs(filtered_img, alpha=a)

    filtered_img = cv2.fastNlMeansDenoisingColored(filtered_img, None, 10, 10, 7, 15)
    filtered_img = cv2.fastNlMeansDenoisingColored(filtered_img, None, 10, 10, 7, 15)

    return filtered_img

if __name__ == '__main__':
    image_path = input("Enter the path to the image: ")
    if not os.path.isfile(image_path):
        print("Invalid image path.")
    else:
        deblurred_image = deblur_image(image_path)
        cv2.imshow("Original Image", cv2.imread(image_path))
        cv2.imshow("Deblurred Image", deblurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
