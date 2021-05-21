import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def pad(array, p):
    array_height = array.shape[0]
    array_width = array.shape[1]
    array_depth = array.shape[2]
    padded_h = array_height + 2*p
    padded_w = array_width + 2*p
    padded_array = np.zeros((padded_h, padded_w, array_depth))
    padded_array[p:padded_h-p,p:padded_w-p,:] = array
    padded_array = np.uint8(padded_array)
    return padded_array

# returns a black and white copy of the image 
def make_gray(array):
    h = array.shape[0]
    w = array.shape[1]
    gray = np.zeros((h,w,3))
    a1 = array[:,:,0] * 1.0
    a2 = array[:,:,1] * 1.0
    a3 = array[:,:,2] * 1.0
    layer = np.add(np.add(a1, a2), a3) / 3
    gray[:,:,0] = layer
    gray[:,:,1] = layer
    gray[:,:,2] = layer
    return np.uint8(gray)

def pass_filter(array, filtr):
    # size parameters
    f_size = filtr.shape[0]
    a_h = array.shape[0]
    a_w = array.shape[1]
    a_d = array.shape[2]

    p = int((f_size -1) / 2) # required padding

    # construct padded array, filter will pass through
    # this array
    padded_array = pad(array, p)

    # create the output array, filled with zeros for now
    output = np.zeros((a_h, a_w, a_d))

    for i in range(a_h):
        for j in range(a_w):
            for k in range(a_d):
                target = padded_array[
                        i:i+f_size,j:j+f_size,k]
                mult = np.multiply(target, filtr[:,:,k])
                summed = round(np.sum(mult))
                output[i,j,k] = summed

            # target = padded_array[i:i+f_size,j:j+f_size,:]
            # multiplied = np.multiply(target, filtr)
            # summed = round(np.sum(multiplied))
            # output[i,j] = summed

    output = np.uint8(output)
    return output

def make_flat(img):
    gray = make_gray(img)
    flat = gray[:,:,0]
    flat = flat / 255
    return flat

def downsample(img, factor):
    i_h = img.shape[0]
    i_w = img.shape[1]

    n_h = round((i_h - factor)/factor + 1)
    n_w = round((i_w - factor)/factor + 1)

    output = np.zeros((n_h, n_w))
    filtr = np.ones((factor,factor)) / factor**2

    for i in range(0, i_h, factor):
        for j in range(0, i_w, factor):
            target = img[i:i+factor, j:j+factor]
            mult = np.multiply(target, filtr)
            summed = np.sum(mult)
            output[int(i/factor),int(j/factor)] = summed

    return output

def cross_correlate(img, temp):
    # CALCULATE OUTPUT MATRIX SIZE
    ### START ###
    # size parameters
    i_h = img.shape[0]   # image height
    i_w = img.shape[1]   # image width
    t_h = temp.shape[0] # template height
    t_w = temp.shape[1] # template width

    n_h = i_h - t_h + 1 # output matrix height
    n_w = i_w - t_w + 1 # output matrix width

    out = np.zeros((n_h, n_w))
    ### END ###

    # CALCULATE AVERAGE VALUE OF TEMPLATE
    ### START ###
    temp_sum = np.sum(temp)
    temp_avg = temp_sum / (t_h * t_w) 
    temp_cov = temp - temp_avg
    temp_cov_sqrd = temp_cov**2
    temp_cov_sqrd_sum = np.sum(temp_cov_sqrd)
    temp_var = np.sqrt(temp_cov_sqrd_sum)
    ### END ###
    
    # PASS THE FILTER AND POPULATE OUTPUT MATRIX
    ### START ###
    for i in range(n_h):
        for j in range(n_w):
            target = img[i:i+t_h, j:j+t_w]
            target_cov = target - np.sum(target)/(t_h * t_w) 
            target_cov_sqrd = target_cov**2
            target_cov_sqrd_sum = np.sum(target_cov_sqrd)
            target_var = np.sqrt(target_cov_sqrd_sum)
            mult = np.multiply(target_cov, temp_cov)
            num = np.sum(mult)
            den = temp_var * target_var
            out[i,j] = num / den
    ### END ###

    return out 

def fast_xCorr(img, temp, factor):
    # template size parameters
    t_h = temp.shape[0]
    t_w = temp.shape[1]

    # 1. DOWNSAMPLE AND CALCULATE SIZE OF OUTPUT
    img_smol = downsample(img, factor)
    temp_smol = downsample(temp, factor)
    is_h = img_smol.shape[0]
    is_w = img_smol.shape[1]
    ts_h = temp_smol.shape[0]
    ts_w = temp_smol.shape[1]
    n_h = is_h - ts_h + 1
    n_w = is_w - ts_w + 1

    # 2. CROSS CORRELATE DOWNSAMPLED IMG AND FIND X AND Y
    out1 = cross_correlate(img_smol, temp_smol)
    coords = np.unravel_index(np.argmax(out1),(n_h, n_w))
    y_1 = coords[0] * factor
    x_1 = coords[1] * factor

    pm = 15

    out2 = np.zeros((pm*2,pm*2))

    # CALCULATE AVERAGE VALUE OF TEMPLATE
    ### START ###
    temp_sum = np.sum(temp)
    temp_avg = temp_sum / (t_h * t_w) 
    temp_cov = temp - temp_avg
    temp_cov_sqrd = temp_cov**2
    temp_cov_sqrd_sum = np.sum(temp_cov_sqrd)
    temp_var = np.sqrt(temp_cov_sqrd_sum)
    ### END ###

    for i in range(y_1 - pm, y_1 + pm):
        for j in range(x_1 - pm, x_1 + pm):
            target = img[i:i+t_h, j:j+t_w]
            target_cov = target - np.sum(target)/(t_h * t_w) 
            target_cov_sqrd = target_cov**2
            target_cov_sqrd_sum = np.sum(target_cov_sqrd)
            target_var = np.sqrt(target_cov_sqrd_sum)
            mult = np.multiply(target_cov, temp_cov)
            num = np.sum(mult)
            den = temp_var * target_var
            out2[i-y_1+pm,j-x_1+pm] = num / den

    coords2 = np.unravel_index(np.argmax(out2), (pm*2,pm*2))
    y_2 = y_1 + coords2[0] - pm
    x_2 = x_1 + coords2[1] - pm
    return y_2, x_2
    
def crop_random(img):
    i_h = img.shape[0]
    i_w = img.shape[1]
    i_d = img.shape[2]

    c_h = round(i_h / 10)
    c_w = round(i_w / 10)

    y = np.random.randint(0, i_h - c_h - 5)
    x = np.random.randint(0, i_w - c_w - 5)

    cropped = img[y:y+c_h, x:x+c_w, :]
    return y, x, cropped

def fast_flat(img):
    img = img / 255
    a1 = img[:,:,0]
    a2 = img[:,:,1]
    a3 = img[:,:,2]
    avg = np.add(np.add(a1, a2), a3) / 3
    return avg

# EXAMPLE OF CROSS CORRELATION
# flat_img = fast_flat(array)
# y, x, crop = crop_random(array)
# flat_crop = fast_flat(crop)
# x2, y2 = fast_xCorr(flat_img, flat_crop, 3)
# print("y-true: ", y)
# print("x-true: ", x)
# print("obtained coordinates: ", (x2, y2))
# f, axarr = plt.subplots(2)
# axarr[0].imshow(flat_img)
# axarr[1].imshow(flat_crop)
# plt.show()

# EXAMPLE OF BLURRING FILTER
# filtr = np.zeros((5,5,3)) + 1/25
# out = pass_filter(array, filtr)
# image2 = Image.fromarray(out)
# 
# f,axarr = plt.subplots(2)
# axarr[0].imshow(image)
# axarr[1].imshow(image2)
# plt.show()
