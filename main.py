from matplotlib import pyplot as plt
import numpy as np

from skimage import io
from skimage.filters import gaussian
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu
from skimage.filters import threshold_local
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import canny
from skimage.color import label2rgb
from skimage.color import rgb2gray
from skimage.transform import resize, hough_line, hough_line_peaks
from skimage.morphology import binary_closing, binary_opening
from skimage.measure import label, regionprops

from scipy.ndimage.morphology import binary_fill_holes


def calculate_avg_door_width(p1, p2, theta1, theta2, d1, d2, need_plot=False):
    eps = 1e-8
    residual_cos = np.cos(theta1) * np.cos(theta2) + np.sin(theta1) * np.sin(theta2)
    if abs(residual_cos) < eps:
        print("Door lines are almost orthogonal")
        return 0.0
    signed_distance1_2 = (d2 - p1[0] * np.cos(theta2) - p1[1] * np.sin(theta2)) / residual_cos
    signed_distance2_1 = (d1 - p2[0] * np.cos(theta1) - p2[1] * np.sin(theta1)) / residual_cos
    if need_plot is True:
        plt.plot((p1[0], p1[0] + signed_distance1_2 * np.cos(theta1)),
                 (p1[1], p1[1] + signed_distance1_2 * np.sin(theta1)), '-g')
        plt.plot((p2[0], p2[0] + signed_distance2_1 * np.cos(theta2)),
                 (p2[1], p2[1] + signed_distance2_1 * np.sin(theta2)), '-g', label='Distance')
        plt.legend(loc='upper right')
    return (abs(signed_distance2_1) + abs(signed_distance1_2)) / 2.0


def select_lines_pair(theta_arr, d_arr):
    # find pairs of approximately parallel lines
    line_pairs_arr = [(theta_i, theta_j, d_i, d_j)
                      for i, (theta_i, d_i) in enumerate(zip(theta_arr, d_arr))
                      for j, (theta_j, d_j) in enumerate(zip(theta_arr, d_arr))
                      if j > i and abs(theta_j - theta_i) < np.pi / 6.0]
    if len(line_pairs_arr) == 0:
        print("Failed to find pair of lines")
        return
    elif len(line_pairs_arr) == 1:
        return line_pairs_arr[0]
    else:
        # if there are several pairs, chose the most "vertical" pair
        return min(line_pairs_arr, key=lambda pair: max(abs(pair[0]), abs(pair[1])))


def find_intersection_points(theta, d, shape):
    x_min, y_min = 0.0, 0.0
    y_max, x_max = shape

    x1, y1 = (d - y_min * np.sin(theta)) / np.cos(theta), y_min
    if x1 < x_min:
        x1, y1 = x_min, (d - x_min * np.cos(theta)) / np.sin(theta)
    elif x1 > x_max:
        x1, y1 = x_max, (d - x_max * np.cos(theta)) / np.sin(theta)

    x2, y2 = (d - y_max * np.sin(theta)) / np.cos(theta), y_max
    if x2 < x_min:
        x2, y2 = x_min, (d - x_min * np.cos(theta)) / np.sin(theta)
    elif x2 > x_max:
        x2, y2 = x_max, (d - x_max * np.cos(theta)) / np.sin(theta)

    return (x1, y1), (x2, y2)


def find_door(image, need_plot=False):
    # binarize the image using otsu
    image_otsu = image <= threshold_otsu(image)

    # apply Hough transform to the image edges
    h, angle, dist = hough_line(canny(image_otsu))
    _, theta, d = hough_line_peaks(h, angle, dist)

    # select the pair of the lines, which most likely is a door
    pair = select_lines_pair(theta, d)
    if pair is None:
        return
    theta1, theta2, d1, d2 = pair

    # find the points of intersection between the lines and the image borders
    point1_1, point1_2 = find_intersection_points(theta1, d1, image.shape)
    point2_1, point2_2 = find_intersection_points(theta2, d2, image.shape)

    if need_plot is True:
        plt.figure()
        plt.title('Finding door width')
        plt.imshow(image, cmap="gray")
        plt.plot((point1_1[0], point1_2[0]), (point1_1[1], point1_2[1]), '-r')
        plt.plot((point2_1[0], point2_2[0]), (point2_1[1], point2_2[1]), '-r', label='Detected lines')
        plt.legend(loc='upper right')

    # calculate the "average" distance between this two lines, it will be treated as the door width
    door_width = calculate_avg_door_width(point1_1, point2_2, theta1, theta2, d1, d2, need_plot)
    return door_width, (theta1, d1), (theta2, d2)


def my_binarization(image, th, color):
    bin_img = [[1 if np.math.dist(pixel, color) < th else 0 for pixel in line] for line in image]
    return bin_img


def get_two_largest_components(segmentation):
    labels = label(segmentation)
    props = regionprops(labels)
    areas = [prop.area for prop in props]
    if len(areas) == 0:
        print("No components found")
        return None, None
    if len(areas) == 1:
        print("Single component found")
        return labels, None
    sorted_areas = sorted(enumerate(areas), key=lambda p: p[1], reverse=True)
    largest_comp_id = sorted_areas[0][0]
    second_largest_comp_id = sorted_areas[1][0]
    return labels == (largest_comp_id + 1), labels == (second_largest_comp_id + 1)


def check_fit_in(segment, door, extension_cff=0.25):
    width, (theta1, d1), (theta2, d2) = door
    not_fit_in = np.zeros_like(segment, dtype=bool)
    result = True

    for y in range(len(segment)):
        for x in range(len(segment[y])):
            if segment[y][x]:
                signed_dist1 = y * np.sin(theta1) + x * np.cos(theta1) - d1
                signed_dist2 = y * np.sin(theta2) + x * np.cos(theta2) - d2
                if signed_dist1 * signed_dist2 > 0.0:
                    if min(abs(signed_dist1), abs(signed_dist2)) > width * extension_cff:
                        not_fit_in[y][x] = True
                        result = False

    return result, not_fit_in


resize_cff = 0.25
brown = [0.22647059, 0.12352941, 0.07352941]
plot_params = {'initial image': False, 'all th': False, 'local th': False, 'otsu + my binary': False,
               'edges maps': False, 'door lines': True, 'result segmentation': True}

img = io.imread("Photoset/1_t.JPG")
initial_h, initial_w, channels = img.shape
img_resized = resize(img, (initial_h * resize_cff, initial_w * resize_cff))
img_gray = rgb2gray(img_resized)
img_gray_blur = gaussian(img_gray, 2)
if plot_params['initial image']:
    plt.imshow(img_resized)
    plt.title('Initial image')

if plot_params['all th']:
    fig_all_th, ax_all_th = try_all_threshold(img_gray_blur, figsize=(15, 15), verbose=False)

img_local = img_gray_blur >= threshold_local(img_gray_blur, 5, method='gaussian', param=25)
if plot_params['local th']:
    fig_th_local, ax_th_local = plt.subplots(1, 4, figsize=(15, 6))
    ax_th_local[0].imshow(img_gray_blur, cmap='gray')
    ax_th_local[0].set_title("Initial grayscale")
    ax_th_local[1].imshow(img_gray_blur >= threshold_local(img_gray_blur, 5, method='mean'), cmap='gray')
    ax_th_local[1].set_title("Local: Mean")
    ax_th_local[2].imshow(img_local, cmap='gray')
    ax_th_local[2].set_title("Local: Gaussian")
    ax_th_local[3].imshow(img_gray_blur >= threshold_local(img_gray_blur, 5, method='median'), cmap='gray')
    ax_th_local[3].set_title("Local: Median")

if plot_params['otsu + my binary']:
    fig_th, ax_th = plt.subplots(1, 3)
    ax_th[0].imshow(img_gray_blur, cmap="gray")
    ax_th[0].set_title("Initial grayscale")
    ax_th[1].imshow(img_gray_blur <= threshold_otsu(img_gray_blur), cmap="gray")
    ax_th[1].set_title("Otsu binary")
    ax_th[2].imshow(my_binarization(img_resized, 0.07, brown), cmap="gray")
    ax_th[2].set_title("My binary")

canny_res = canny(img_gray_blur, sigma=1.6)
if plot_params['edges maps']:
    fig_edges, ax_edges = plt.subplots(2, 3, figsize=(15, 6))
    ax_edges[0, 0].imshow(img_gray_blur, cmap='gray')
    ax_edges[0, 1].imshow(canny_res, cmap='gray')
    ax_edges[0, 2].imshow(roberts(img_gray_blur), cmap='gray')
    ax_edges[1, 0].imshow(sobel(img_gray_blur), cmap='gray')
    ax_edges[1, 1].imshow(scharr(img_gray_blur), cmap='gray')
    ax_edges[1, 2].imshow(prewitt(img_gray_blur), cmap='gray')
    for i, title in enumerate(["Input grayscale", "Canny", "Roberts", "Sobel", "Scharr", "Prewitt"]):
        ax_edges.flatten()[i].set_title(title)

edge_map = binary_closing(canny_res, selem=np.ones((21, 21)))
edge_segmentation = binary_fill_holes(edge_map)
edge_segmentation_opened = binary_opening(edge_segmentation, selem=np.ones((3, 3)))

answer = None
result_segmentation_img = None
door = find_door(img_gray_blur, need_plot=plot_params['door lines'])
if door is None:
    door = find_door(edge_segmentation_opened, need_plot=plot_params['door lines'])
if door is None:
    print("Failed to find door")
else:
    comp1, comp2 = get_two_largest_components(edge_segmentation_opened)
    if comp1 is None:
        print("Failed to find largest comp")
    else:
        answer = "The chair fits in"
        result, not_fit_in = check_fit_in(comp1, door, 0.25)
        result_segmentation_img = label2rgb(comp1, image=img_gray_blur, colors=['white', 'blue'])
        # if the largest component doesn't fit in
        if not result:
            result_segmentation_img = label2rgb(not_fit_in, image=img_gray_blur, colors=['white', 'red'])
            answer = "The chair doesn't fit in"
        # else check the second largest component
        else:
            print("Check second component")
            if comp2 is not None:
                props1 = regionprops(label(comp1))
                props2 = regionprops(label(comp2))
                area1 = props1[0].area
                area2 = props2[0].area
                perimeter2 = props2[0].perimeter
                # check if the second component isn't too small and has appropriate shape
                if area1 / area2 > 50.0:
                    print("The second component has too small area")
                elif area2 / perimeter2 < 10.0:
                    print("The second component has too elongated shape")
                else:
                    result, not_fit_in = check_fit_in(comp2, door)
                    if not result:
                        result_segmentation_img = label2rgb(not_fit_in, image=img_gray_blur, colors=['white', 'red'])
                        answer = "The chair doesn't fit in"

if plot_params['result segmentation']:
    if answer is None:
        fig_seg, ax_seg = plt.subplots(1, 3, figsize=(15, 6))
    else:
        fig_seg, ax_seg = plt.subplots(1, 4, figsize=(15, 6))
    ax_seg[0].imshow(img_gray_blur, cmap='gray')
    ax_seg[0].set_title("Initial grayscale")
    ax_seg[1].imshow(label2rgb(edge_segmentation, image=img_gray_blur, colors=['white', 'blue']))
    ax_seg[1].set_title("Edge-based segmentation")
    ax_seg[2].imshow(label2rgb(edge_segmentation_opened, image=img_gray_blur, colors=['white', 'blue']))
    ax_seg[2].set_title("After bin. opening")
    if answer is not None:
        ax_seg[3].imshow(result_segmentation_img)
        ax_seg[3].set_title(answer)

print("Answer: " + str(answer))
plt.show()


