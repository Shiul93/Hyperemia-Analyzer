import cv2
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Eye redness analysis')
parser.add_argument('input', metavar='F', type=str,
                    help='Input file route')

args = parser.parse_args()

# input_path = "dataset/alto/BN_056_3.jpg"
# input_path = "dataset/medio/BT_13_2.BMP"
# input_path = "dataset/bajo/BN_11_1.5.jpg"

input_path = args.input

match = re.match(".*?/([A-Za-z_0-9]+)?_([0-9\.]+)\.", input_path)



# ---------------------------------------------------FUNCTIONS----------------------------------------------------------#

def horizontalVascularComponent(image, threshold=0.97, maxval=255):
    bluechan = image[:, :, 1] * 1
    bluechan_norm = bluechan * 1
    output = bluechan * 1
    cv2.normalize(bluechan, bluechan_norm, 0, 255, cv2.NORM_MINMAX)
    
    mean = bluechan_norm.mean()
    print mean
    meanthreshold = round(mean * threshold)
    rows, cols = bluechan_norm.shape
    for i in xrange(rows):
        for j in xrange(cols):
            if bluechan_norm[i, j] < meanthreshold:
                output[i, j] = maxval
            else:
                output[i, j] = 0

    

    return output


def adaptiveThreshold(img, thresholdType=cv2.THRESH_BINARY_INV):
    imcopy = np.zeros(img.shape)
    mask = cv2.adaptiveThreshold(img[:, :, 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 thresholdType, 9, 2)

    dilated = cv2.dilate(mask, np.ones((1, 1)))
    erode = cv2.erode(dilated, np.ones((1, 1)))
    mask = erode

    return mask


def meanValuesMasked(img, mask):
    b, g, r = cv2.split(img)
    comb = r - (g * 0.83) - (b * 0.17)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    nonZero = cv2.countNonZero(mask)
    acum_r = 0
    acum_h = 0
    acum_s = 0
    acum_v = 0
    for idx, row in enumerate(mask):
        for idy, x in enumerate(row):
            if x != 0:
                acum_r += comb[idx][idy]
                acum_h += h[idx][idy]
                acum_s += s[idx][idy]
                acum_v += v[idx][idy]

    mean_r = acum_r / float(nonZero) / float(255)
    mean_h = acum_h / float(nonZero) / float(255)
    mean_s = acum_s / float(nonZero) / float(255)
    mean_v = acum_v / float(nonZero) / float(255)

    return [mean_r, mean_h, mean_s, mean_v]


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calculateRepresentativeVector(img_in):
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    out = np.true_divide(img.sum(1), (img != 0).sum(1))
    return [out[:, 0].mean(), out[:, 1].mean(), out[:, 2].mean()]


def distanceThreshold(img_in, representativevector, threshold):
    output = np.zeros(img_in.shape)
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)

    rows, cols, chans = img.shape
    for x in range(rows):
        for y in range(cols):
            if angle_between(img[x, y, :], representativevector) < threshold:
                output[x, y, :] = img_in[x, y, :]

    return output


# -----------------------------------------------------MAIN CODE--------------------------------------------------------#


def on_mouse_clicked(event, x, y, flags, params):
    # print event, x, y, flags, params
    show = True
    img_copy = params["image"] * 1
    # Primer click
    if event == cv2.EVENT_LBUTTONDOWN:
        # print "click en", x, y," valor ",img_hsv[x,y]
        params["x1"] = x
        params["y1"] = y
        params["roi"] = None
        cv2.circle(img_copy, (x, y), 5, (0, 255, 255), 2)
    # Movimiento
    elif event == cv2.EVENT_MOUSEMOVE:
        # print "moviendonos por", x, y
        if params["x1"]:
            # cv2.putText(img_copy, 'HSV: ' + str(img_hsv[x, y]), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            # cv2.putText(img_copy, 'COORDS: ' + str([x, y]), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.circle(img_copy, (params["x1"], params["y1"]), 5, (0, 255, 255), 2)
            cv2.rectangle(img_copy, (params["x1"], params["y1"]), (x, y), (0, 255, 255), 2)
        else:
            show = False
    # Segundo click
    elif event == cv2.EVENT_LBUTTONUP:
        # print "dejando de clickar en", x, y

        # cv2.putText(img_copy, 'HSV: ' + str(img_copy[x, y]), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


        # cv2.putText(img_copy, 'COORDS: ' + str([x, y]), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


        cv2.rectangle(img_copy, (params["x1"], params["y1"]), (x, y), (255, 255, 0), 2)

        x1 = params["x1"]
        y1 = params["y1"]
        params["roi"] = [x1, y1, x, y]
        if x1 < x:
            if y1 < y:
                params["roi_img"] = [y1, y, x1, x]

            else:
                params["roi_img"] = [y, y1, x1, x]

        else:
            if y1 < y:
                params["roi_img"] = [y1, y, x, x1]

            else:
                params["roi_img"] = [y, y1, x, x1]

        roi = params["roi_img"]

        if (x1 == x) and (y1 == y):
            cv2.putText(img_copy, 'POINT: [' + str(x) + ', ' + str(y) + ']', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0))

        else:
            cv2.putText(img_copy, 'ROI: ' + str(roi), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        params["x1"] = None
        params["x2"] = None

    if show:
        cv2.imshow("window", img_copy)

    if flags == cv2.EVENT_FLAG_CTRLKEY:
        print "pulsando tecla ctrl"


im = cv2.imread(input_path)
img_hsv = cv2.cvtColor(im * 1, cv2.COLOR_BGR2HSV)

cv2.namedWindow("window")

params = dict(image=im, x1=None, y1=None, roi=None, roi_img=None, imregion=None, imregion_hsv=None)
cv2.setMouseCallback("window", on_mouse_clicked, params)
cv2.setMouseCallback("regionproc", on_mouse_clicked, params)

stop = False
while not stop:
    cv2.imshow("window", im)
    k = cv2.waitKey() & 0xff
    if k == ord('q'):
        stop = True
    elif k == ord('0'):
        print("Pulsada tecla 0")
        imregion_copy = params['imregion'] * 1
        cv2.cvtColor(params['imregion'], cv2.COLOR_BGR2HSV, imregion_copy)
        params['imregion_hsv'] = imregion_copy
        cv2.imshow("regionproc", imregion_copy)






    elif k == ord('w'):
        # print("Pulsada tecla w")
        # region = im[params["roi"][1]:params["roi"][3],params["roi"][0]:params["roi"][2]]
        region = im[params["roi_img"][0]:params["roi_img"][1], params["roi_img"][2]:params["roi_img"][3]]
        params['imregion'] = region


    elif k == ord('a'):
        print ""
        print "--------------------------------------------"
        print input_path
        print("---------- Classification metrics ----------")
        imcop = np.zeros(params['imregion'][:, :, 1].shape)
        imori = params['imregion']

        b, g, r = cv2.split(imori)
        # General red level
        imcop = r - (g * 0.83) - (b * 0.17)
        general_redlevel = imcop.mean()
        print "General red level " + str(imcop.mean() / float(255))
        # Vascular red & saturation level
        mask = adaptiveThreshold(imori)
        vascular_area = float(np.count_nonzero(mask)) / float(mask.size)
        print "Vascular area " + str(vascular_area)
        mean_redlevel, mean_hue, mean_saturation, mean_value = meanValuesMasked(imori, mask)
        print "_____Vascular region metrics_____"
        print "Red level at vascular section " + str(mean_redlevel)
        print "Hue level at vascular section " + str(mean_hue)
        print "Saturation level at vascular section " + str(mean_saturation)
        print "Value level at vascular section " + str(mean_value)
        # Bulbar red & saturation level


        mask = adaptiveThreshold(imori, cv2.THRESH_BINARY)
        mean_redlevel2, mean_hue2, mean_saturation2, mean_value2 = meanValuesMasked(imori, mask)
        print "_____Bulbar region metrics_____"

        print "Red level at Bulbar section " + str(mean_redlevel2)
        print "Hue level at Bulbar section " + str(mean_hue2)
        print "Saturation level at Bulbar section " + str(mean_saturation2)
        print "Value level at Bulbar section " + str(mean_value2)
        print "_____CSV OUTPUT_____"
        print str(general_redlevel) + "," + str(mean_redlevel) + "," + str(mean_hue) + "," + str(mean_saturation)+ "," + str(
            mean_value) + "," + str(mean_redlevel2) + "," + str(mean_hue2) + "," + str(mean_saturation2)+ "," + str(mean_value2) + "," + str(
            vascular_area) + "," + match.group(2)

        #prediction = 0.2921449577 * general_redlevel - 14.35255528 * mean_redlevel + 17.70632605 * mean_hue - 28.75683329 * mean_value - 48.36114162 * mean_redlevel2 - 18.10658115 * mean_hue2 + 28.22720168 * mean_value2 - 3.363273212 * vascular_area + 1.31451989
        prediction =  0.0291 * general_redlevel -46.105  *  mean_hue +6.3641 *  mean_saturation -2.0444 *  mean_value +10.6774 *  mean_hue2 -6.0809 *  mean_saturation2 -1.1733 *  vascular_area + 3.5845
        print "Predicted hyperemia value: " + str(prediction)

    elif k == ord('c'):

        imcop = np.zeros(params['imregion'][:, :, 1].shape)
        imori = params['imregion']
        b, g, r = cv2.split(imori)
        # General red level
        imcop = r - (g * 0.83) - (b * 0.17)
        general_redlevel =imcop.mean()

        # Vascular red & saturation level
        mask = adaptiveThreshold(imori)
        vascular_area = float(np.count_nonzero(mask)) / float(mask.size)

        mean_redlevel, mean_hue, mean_saturation, mean_value = meanValuesMasked(imori, mask)

        mask = adaptiveThreshold(imori, cv2.THRESH_BINARY)
        mean_redlevel2, mean_hue2, mean_saturation2, mean_value2 = meanValuesMasked(imori, mask)

        print str(general_redlevel) + "," + str(mean_redlevel) + "," + str(mean_hue) + "," + str(
            mean_saturation) + "," + str(
            mean_value) + "," + str(mean_redlevel2) + "," + str(mean_hue2) + "," + str(mean_saturation2) + "," + str(
            mean_value2) + "," + str(
            vascular_area) + "," + match.group(2)

    elif k == ord('p'):

        imcop = np.zeros(params['imregion'][:, :, 1].shape)
        imori = params['imregion']
        b, g, r = cv2.split(imori)
        # General red level
        imcop = r - (g * 0.83) - (b * 0.17)
        general_redlevel = imcop.mean()

        # Vascular red & saturation level
        mask = adaptiveThreshold(imori)
        vascular_area = float(np.count_nonzero(mask)) / float(mask.size)

        mean_redlevel, mean_hue, mean_saturation, mean_value = meanValuesMasked(imori, mask)

        mask = adaptiveThreshold(imori, cv2.THRESH_BINARY)
        mean_redlevel2, mean_hue2, mean_saturation2, mean_value2 = meanValuesMasked(imori, mask)

        prediction =  0.0291 * general_redlevel -46.105  *  mean_hue +6.3641 *  mean_saturation -2.0444 *  mean_value +10.6774 *  mean_hue2 -6.0809 *  mean_saturation2 -1.1733 *  vascular_area + 3.5845
        print str(prediction)+", "+ match.group(2)

    elif k == ord('t'):
        print("Adaptative threshold")
        imcopy = params['imregion'] * 1
        imcop = cv2.adaptiveThreshold(params['imregion'][:, :, 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 9, 2)

        dilated = cv2.dilate(imcop, np.ones((1, 1)))
        erode = cv2.erode(dilated, np.ones((1, 1)))

        # dilated = cv2.dilate(imcop, np.ones((3, 3)))

#        print imcopy
        imcopy[:, :, 0] = imcopy[:, :, 0] * (erode / 255)
        imcopy[:, :, 1] = imcopy[:, :, 1] * (erode / 255)
        imcopy[:, :, 2] = imcopy[:, :, 2] * (erode / 255)

        cv2.imshow("regionproc", imcopy)

    elif k == ord('h'):

        print "general_red_level, vas_mean_redlvl, vas_mean_hue, vas_mean_sat, vas_mean_val, bulb_mean_redlvl, bulb_mean_hue, bulb_mean_sat, bulb_mean_val, vascular_area, expert_lvl"




        # im[params["roi2"]] = cv2.cvtColor(im[params["roi2"]]*1,cv2.COLOR_HSV2BGR)
