import torch
import numpy as np
from utils.visual.visual import load_image, to_image
from matplotlib import pyplot as plt


def fft_img(img1, img2, img3):
    fft_1 = np.zeros(32,)
    fft_2 = np.zeros(32,)
    fft_3 = np.zeros(32,)
    for i in range(32):
        fft1 = (torch.fft.fft(img1[0][i]) + torch.fft.fft(img1[1][i]) + torch.fft.fft(img1[2][i]))/1
        fft2 = (torch.fft.fft(img2[0][i]) + torch.fft.fft(img2[1][i]) + torch.fft.fft(img2[2][i]))/1
        fft3 = (torch.fft.fft(img3[0][i]) + torch.fft.fft(img3[1][i]) + torch.fft.fft(img3[2][i]))/1
        fft1 = (fft1 * torch.conj(fft1)).real
        fft2 = (fft2 * torch.conj(fft2)).real
        fft3 = (fft3 * torch.conj(fft3)).real
        
        fft_1 += np.array(fft1)
        fft_2 += np.array(fft2)
        fft_3 += np.array(fft3)
        # print(list(np.array(fft1)))
        # print(list(np.array(fft2)))
        # print(list(np.array(fft3)))
        # for key in range(len(fft1)):
        #     if torch.abs(fft2[key].real - fft1[key].real) > 3 or torch.abs(fft2[key].imag - fft1[key].imag) > 3 or \
        #         torch.abs(fft3[key].real - fft1[key].real) > 3 or torch.abs(fft3[key].imag - fft1[key].imag) > 3:
        #         print(key)
        
        
        # input("Press Enter to continue...")
    print(list(fft_1[:8]))
    print(list(fft_2[:8]))
    print(list(fft_3[:8]))
    print("___________________________________")
    print(list(fft_1[24:31]))
    print(list(fft_2[24:31]))
    print(list(fft_3[24:31]))
    pass

def fft2_img(img1, i):
    img1 = torch.fft.fft2(img1)
    img1 = torch.fft.fftshift(img1)
    img1 = (img1 * torch.conj(img1)).real
    # magnitude_spectrum = 20*np.log(np.abs(img1))

    # plt.subplot(121),plt.imshow(img1, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(magnitude_spectrum.reshape(32, 32, 3), cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.savefig( "images/fft2_test_grad_spectrum" + str(i) + ".jpg")
    # plt.show()
    to_image(img1, "images/fft2_test_grad" + str(i) + ".jpg", img_size=32)



if __name__ == "__main__":
    
    idx = 1
    img1 = load_image("images/cifar_test" + str(idx) + ".jpg")
    img2 = load_image("images/cifar_test" + str(idx) + "_adv_swin.jpg")
    img3 = load_image("images/cifar_test" + str(idx) + "_adv_wideresnet.jpg")
    # fft2_img(img1, 1)
    fft2_img(img2 - img1, 211)
    fft2_img(img3 - img1, 311)