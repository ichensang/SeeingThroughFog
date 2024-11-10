import numpy as np
import os
import json
import cv2
from pyquaternion import Quaternion
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from fuzzy_algocompare import *
import xml.etree.ElementTree as ET

def find_image_by_name(options,filename):
    # Iterate through each <image> element
    for image in options:
        if image.get('name') == filename:
            return image
    return None


def metric_compute(testimg, gtimg):
    croptest = testimg[600:810,:].flatten()
    cropgt = gtimg[600:810,:].flatten()
    intersection = np.sum(croptest*cropgt)
    union = np.sum(croptest) + np.sum(cropgt) - intersection
    iou = intersection/union
    f1 = 2*intersection/(np.sum(croptest) + np.sum(cropgt))

    return iou, f1

def load_calib_data(path_total_dataset, name_camera_calib, tf_tree, velodyne_name='lidar_hdl64_s3_roof'):
    assert velodyne_name in ['lidar_hdl64_s3_roof', 'lidar_vlp32_roof'], 'wrong frame id in tf_tree for velodyne_name'

    with open(os.path.join(path_total_dataset, name_camera_calib), 'r') as f:
        data_camera = json.load(f)

    with open(os.path.join(path_total_dataset, tf_tree), 'r') as f:
        data_extrinsics = json.load(f)

    calib_dict = {
        'calib_cam_stereo_left.json': 'cam_stereo_left_optical',
        'calib_cam_stereo_right.json': 'cam_stereo_right_optical',
        'calib_gated_bwv.json': 'bwv_cam_optical'
    }

    cam_name = calib_dict[name_camera_calib]

    important_translations = [velodyne_name, 'radar', cam_name]
    translations = []

    for item in data_extrinsics:
        if item['child_frame_id'] in important_translations:
            translations.append(item)
            if item['child_frame_id'] == cam_name:
                T_cam = item['transform']
            elif item['child_frame_id'] == velodyne_name:
                T_velodyne = item['transform']
    R_c_quaternion = Quaternion(w=T_cam['rotation']['w'] * 360 / 2 / np.pi, x=T_cam['rotation']['x'] * 360 / 2 / np.pi,
                                y=T_cam['rotation']['y'] * 360 / 2 / np.pi, z=T_cam['rotation']['z'] * 360 / 2 / np.pi)
    R_v_quaternion = Quaternion(w=T_velodyne['rotation']['w'] * 360 / 2 / np.pi,
                                x=T_velodyne['rotation']['x'] * 360 / 2 / np.pi,
                                y=T_velodyne['rotation']['y'] * 360 / 2 / np.pi,
                                z=T_velodyne['rotation']['z'] * 360 / 2 / np.pi)

    # Setup quaternion values as 3x3 orthogonal rotation matrices
    R_c_matrix = R_c_quaternion.rotation_matrix
    R_v_matrix = R_v_quaternion.rotation_matrix

    Tr_cam = np.asarray([T_cam['translation']['x'], T_cam['translation']['y'], T_cam['translation']['z']])
    Tr_velodyne = np.asarray(
        [T_velodyne['translation']['x'], T_velodyne['translation']['y'], T_velodyne['translation']['z']])

    zero_to_camera = np.zeros((3, 4))
    zero_to_camera[0:3, 0:3] = R_c_matrix
    zero_to_camera[0:3, 3] = Tr_cam
    zero_to_camera = np.vstack((zero_to_camera, np.array([0, 0, 0, 1])))

    zero_to_velodyne = np.zeros((3, 4))
    zero_to_velodyne[0:3, 0:3] = R_v_matrix
    zero_to_velodyne[0:3, 3] = Tr_velodyne
    zero_to_velodyne = np.vstack((zero_to_velodyne, np.array([0, 0, 0, 1])))

    velodyne_to_camera = np.matmul(np.linalg.inv(zero_to_camera), zero_to_velodyne)
    camera_to_velodyne = np.matmul(np.linalg.inv(zero_to_velodyne), zero_to_camera)

    P = np.reshape(data_camera['P'], [3, 4])
    R = np.identity(4)
    vtc = np.matmul(np.matmul(P, R), velodyne_to_camera)

    print('vtc = ',vtc)
    print('velodyne_to_camera = ', velodyne_to_camera)

    return velodyne_to_camera, camera_to_velodyne, P, R, vtc, zero_to_camera


def load_velodyne_scan(file):
    """Load and parse velodyne binary file"""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 5))  # [:, :4]


def filter(lidar_data, distance):
    """
    Takes lidar Pointcloud as ibnput and filters point below distance threshold
    :param lidar_data: Input Pointcloud
    :param distance: Minimum distance for filtering
    :return: Filtered Pointcloud
    """

    r = np.sqrt(lidar_data[:, 0] ** 2 + lidar_data[:, 1] ** 2 + lidar_data[:, 2] ** 2)
    true_idx = np.where(r > distance)

    lidar_data = lidar_data[true_idx, :]

    return lidar_data[0]


def find_missing_points(last, strongest):
    last_set = set([tuple(x) for x in last])
    strong_set = set([tuple(x) for x in strongest])
    remaining_last = np.array([x for x in last_set - strong_set])
    remaining_strong = np.array([x for x in strong_set - last_set])

    return remaining_last, remaining_strong



def py_func_project_3D_to_2D(points_3D, P):
    # Project on image
    points_2D = np.matmul(P, np.vstack((points_3D, np.ones([1, np.shape(points_3D)[1]]))))

    # scale projected points
    points_2D[0][:] = points_2D[0][:] / points_2D[2][:]
    points_2D[1][:] = points_2D[1][:] / points_2D[2][:]

    points_2D = points_2D[0:2]
    return points_2D.transpose()


root = 'dataset/'
lidar_type = 'lidar_hdl64'
#photo_root = 'dataset/rainy/image'
photo_root = 'dataset/snowy/street_snow_cover'

ground_truth = ET.parse('snow_street.xml')
root_gt = ground_truth.getroot()
image_elem = root_gt.findall('.//image')
total_iou = 0.0
total_f1 = 0.0

velodyne_to_camera, camera_to_velodyne, P, R, vtc, zero_to_camera = load_calib_data(
    root, name_camera_calib='calib_cam_stereo_left.json', tf_tree='calib_tf_tree_full.json',
    velodyne_name='lidar_hdl64_s3_roof' if lidar_type == 'lidar_hdl64' else 'lidar_vlp32_roof')

velodyne_to_camera2 = np.zeros((5, 5))
velodyne_to_camera2[0:4, 0:4] = velodyne_to_camera
velodyne_to_camera2[4, 4] = 1

print('lidar to cam = ',velodyne_to_camera)
print('P = ', P)

weather_samples = [f for f in os.listdir(photo_root) if os.path.isfile(os.path.join(photo_root, f))]
frame_cnt = len(weather_samples)

interesting_samples = [
    '2018-02-06_14-25-51_00400',
    '2019-09-11_16-39-41_01770',
    '2018-02-12_07-16-32_00100',
    '2018-10-29_16-42-03_00560',
]

echos = [
    ['last', 'strongest'],
]

#CAMERA DEFAULTS===========================================================================
cannythreshold = 20 #20snow 50rain
cannycontrol = 0
left_corner = 0
right_corner = 1920
default_rho_l = -150
default_rho_r = 1000
default_theta_l = -1.25
default_theta_r = 0.96
draw_line_number = 100

#LIDAR DEFAULTS===========================================================================
left_range = 30
right_range = 30
forward_range = 40
grid_size = 0.5
gridmap_y = np.int8(np.ceil(forward_range / grid_size))
gridmap_x = np.int8(np.ceil((left_range + right_range) / grid_size))
NUM_COLOURS = 7
rainbow = [
    [0, 0, 255],  # Red
    [0, 127, 255],  # Orange
    [0, 255, 255],  # Yellow
    [0, 255, 0],  # Green
    [255, 0, 0],  # Blue
    [130, 0, 75],  # Indigo
    [211, 0, 148]  # Violet
]

for sample in weather_samples:
    print('filename = ', sample)
    #image_filename = os.path.join(root, 'cam_stereo_left_lut', interesting_sample + '.png')
    #image_filename = os.path.join(root, 'rainy', 'image', sample)
    image_filename = os.path.join(root, 'snowy', 'street_snow_cover', sample)
    #image_filename = os.path.join(root, 'snowy', 'highway', sample)
    #image_filename = 'dataset/rainy/image/2019-05-02_20-17-41_00510.png'
    #velo_file_last = os.path.join(root, lidar_type + '_' + echos[0][0],
    #                              interesting_sample + '.bin')
    #velo_file_strongest = os.path.join(root, lidar_type + '_' + echos[0][1],
    #                                   interesting_sample + '.bin')
    pointcloud_file = os.path.splitext(sample)[0] + '.bin'
    lidar_filename = os.path.join(root, lidar_type + '_' + echos[0][0], pointcloud_file)


    #read
    img = cv2.imread(image_filename)
    IMG_H, IMG_W, _ = img.shape
    #lidar_data_last = load_velodyne_scan(velo_file_last)
    #lidar_data_last = lidar_data_last[:,0:4]
    #lidar_data_strongest = load_velodyne_scan(velo_file_strongest)
    #lidar_data_strongest = lidar_data_strongest[:,0:4]
    lidar_data_strongest = load_velodyne_scan(lidar_filename)
    lidar_data_strongest = lidar_data_strongest[:, 0:4]
    #cannythreshold = 5
    #CAMERA=============================================================================================================
    for j in range(1):
        cannythreshold = cannythreshold + cannycontrol
        #cannythreshold = 30
        print("canny threshold = ", cannythreshold)
        x_intercept = IMG_W / 2
        y_intercept = IMG_H / 2.1
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(out, 15, 50, 50)
        high = cannythreshold
        low = high / 3
        edge = cv2.Canny(filtered, low, high, None, 3)
        edge = np.uint8(edge)

        roi_h = x_intercept
        roi_v = y_intercept
        # myROI = np.array([[(roi_h, roi_v-30), (0, IMG_H-320), (IMG_W-1, IMG_H-320)]], dtype = np.int32)
        myROI = np.array([[(roi_h - 270, roi_v + 80), (roi_h + 270, roi_v + 80), (right_corner + 800, IMG_H-100),
                           (left_corner, IMG_H-100)]], dtype=np.int32)
        mask = np.zeros_like(edge)
        region = cv2.fillPoly(mask, myROI, 255)
        roi = cv2.bitwise_and(edge, region)

        #filter_show = filtered[::2,::2]
        #cv2.imshow('edge', filter_show)
        #cv2.imshow('roi', roi3)
        #cv2.waitKey(10)

        line_image = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        lines = cv2.HoughLines(roi, 1, np.pi / 180, 3, None, 0, 0)


        if lines is not None:
            rhoall = lines[:, :, 0]
            thetaall = lines[:, :, 1]
            totalinesall = len(rhoall) * 2
            if totalinesall > 70000:
                totalinesall = 70000
            print(totalinesall)
            cannycontrol = fuzzy_canny(totalinesall)
            if draw_line_number > 200: ## downsize original 200
                draw_line_number = 200 ##downsize original 200
            if draw_line_number < 25: ## downsize original 50
                draw_line_number = 25 ##down size original 50
            #print('draw line number = ', draw_line_number)
            draw_line_number = 200
            left = []
            leftrho = []
            right = []
            rightrho = []
            for i in (range(draw_line_number)):
                if (len(right)==3) and (len(left)==3):
                    break
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                if theta > 1.5708:
                    theta = theta-3.14159
                    rho = -rho
                thetad = theta/3.14159*180
                if (thetad<80) & (thetad>20) & (len(right)<3) & (rho>500) & (rho<1200): #55->20
                    right.append(theta)
                    rightrho.append(rho)
                        #print(rho)
                        #print(thetad)
                if (thetad<0) & (thetad>-80) & (len(left)<3) & (rho>-450) & (rho<1200): #50->10 80->90
                    left.append(theta)
                    leftrho.append(rho)
            if len(left) == 0:
                print('None Left')
                averageleft = float(default_theta_l)
                rholeftmean = float(default_rho_l)
                #print(rholeftmean)
                draw_line_number = draw_line_number + 20  ##down size original 10
            else:
                averageleft = np.average(left)
                rholeftmean = np.average(leftrho)
                #default_theta_l = averageleft
                #default_rho_l = rholeftmean
            if len(right) == 0:
                print('None right')
                averageright = float(default_theta_r)
                rhorightmean = float(default_rho_r)
                draw_line_number = draw_line_number + 20  ##down size original 10
            else:
                averageright = np.average(right)
                rhorightmean = np.average(rightrho)
                #default_theta_r = averageright
                #default_rho_r = rhorightmean
            if (len(right) != 0) & (len(left) != 0):
                draw_line_number = draw_line_number - 10

            #averageleft = -70/180*3.14159
            #rholeftmean = -200
            cl = np.cos(averageleft)
            sl = np.sin(averageleft)
            x0l = cl * rholeftmean
            y0l = sl * rholeftmean
            pt1l = ((round((rholeftmean - IMG_H * sl) / cl)), IMG_H)
            pt2l = ((round((rholeftmean - 0.5 * IMG_H * sl) / cl)), round(0.5 * IMG_H))
            cv2.line(line_image, pt1l, pt2l, (0, 255, 255), 2)  ### down size original 13
            #cv2.line(roi, pt1l, pt2l, (0, 255, 255), 2)

            #averageright = 50/180*2.14159
            #rhorightmean = 1100
            cr = np.cos(averageright)
            sr = np.sin(averageright)
            x0r = cr * rhorightmean
            y0r = sr * rhorightmean
            pt1r = ((round((rhorightmean - IMG_H * sr) / cr)), IMG_H)
            pt2r = ((round((rhorightmean - 0.5 * IMG_H * sr) / cr)), round(0.5 * IMG_H))
            cv2.line(line_image, pt1r, pt2r, (0, 255, 255), 2)
            #cv2.line(roi, pt1r, pt2r, (0, 255, 255), 2)
            y_intercept = round((rhorightmean - rholeftmean * cr / cl) / (sr - sl * cr / cl))
        else:
            print("No line")
        #color_edge = np.dstack((result, result, result))
        color_edge = np.dstack((roi,roi,roi))
        color_edge_show = color_edge[::2,::2]
        combo = cv2.addWeighted(color_edge, 0.8, line_image, 1, 0)
        comboresize = combo[::2,::2] ## down size
        cv2.imshow("Combo", comboresize)
        cv2.waitKey(10)

    #LIDAR==============================================================================================================
    #data 3d matrix construction
    lidar_points_3D = np.ones((lidar_data_strongest.shape[0], lidar_data_strongest.shape[1] + 1))
    lidar_points_3D[:, 0:3] = lidar_data_strongest[:, 0:3]
    lidar_points_3D[:, 4] = lidar_data_strongest[:, 3]

    #turn into camera coordinate
    pts_3D = np.matmul(velodyne_to_camera2, lidar_points_3D.transpose())
    pts_3D = np.delete(pts_3D, 3, axis=0)

    within_width = np.logical_and(left_range > pts_3D[0, :], pts_3D[0, :] >-right_range)
    within_distance = np.logical_and(forward_range > pts_3D[2, :], pts_3D[2, :] >= 4)
    valid_points = np.logical_and(within_width, within_distance)
    coordinates = np.where(valid_points)[0]
    lidar_delete_back = pts_3D[:, coordinates] #camera coordinate within range
    lidar_delete_back2 = lidar_data_strongest[coordinates,:] #original coordinate within range

    #calculate image coordinate
    lidar_points_2D = py_func_project_3D_to_2D(lidar_delete_back2[:, 0:3].transpose(), vtc)
    edge_map_lidar2 = np.zeros((2, IMG_W), dtype=np.int16)
    within_image_border_width = np.logical_and(IMG_W > lidar_points_2D[:, 0], lidar_points_2D[:, 0] >= 0)
    within_image_border_height = np.logical_and(IMG_H > lidar_points_2D[:, 1], lidar_points_2D[:, 1] >= 0)
    valid_points_2d = np.logical_and(within_image_border_width, within_image_border_height)
    coordinates_2d = np.where(valid_points_2d)[0]
    img_coordinates = lidar_points_2D[coordinates_2d, :].astype(dtype=np.int32)
    lidar_delete_back3 = lidar_delete_back[:,coordinates_2d] #3d coordinates with range

    #calculate grid  map
    gridmap = np.zeros((gridmap_y, gridmap_x), dtype=float)
    lowest = np.zeros((gridmap_y, gridmap_x), dtype=np.int16)
    grid_coords = np.zeros((2, lidar_delete_back3.shape[1]), dtype=np.int16)
    new_test_map = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    grid_coords[0, :] = gridmap_y - np.int16(np.floor(lidar_delete_back3[2, :] / grid_size)) - 1  # y
    grid_coords[1, :] = np.int16(np.floor((lidar_delete_back3[0, :] + right_range) / grid_size))  # x
    np.add.at(gridmap, (grid_coords[0, :], grid_coords[1, :]), 1)
    for i in range(gridmap_y):
        for j in range(gridmap_x):
            gridmap[i,j] = gridmap[i,j]*((40-0.5*i)*(40-0.5*i)+(j-gridmap_x/2)*(j-gridmap_x/2)*0.25)/2500*5

    for i in range(grid_coords.shape[1]):
        par = gridmap[grid_coords[0, i], grid_coords[1, i]]
        #par = 10
        done = [img_coordinates[i, 0], img_coordinates[i, 1]]
        if par > 8: #27
            colour = 2 #yellow points
            new_test_map[int(done[1]), int(done[0])] = 1
        else:
            colour = 4 #blue
        cv2.circle(img, (int(done[0]), int(done[1])), 0, rainbow[colour], thickness=2, lineType=8, shift=0)
    showimg = img[::2,::2]
    cv2.imshow('tesn', showimg)
    cv2.waitKey(10)

    filter_n = np.ones((1, 25))
    convoluted = convolve2d(new_test_map, filter_n)
    for i in range(IMG_W):
        column = convoluted[600:810, i] ##550 850 change
        non_zero_indices = np.nonzero(column)[0]
        if len(non_zero_indices) <= 3:
            continue
        edge_map_lidar2[0, i] = 600 ##change
        edge_map_lidar2[1, i] = non_zero_indices[-1] + 600 ##change

    for i in range(IMG_W):
        cv2.line(img, (i, edge_map_lidar2[0, i]), (i, edge_map_lidar2[1, i]), [0, 127, 255], thickness=1)

    #cv2.circle(img, (640, 520), 10, [0, 0, 255], thickness=2, lineType=8, shift=0)
    display1 = img[::2, ::2]
    cv2.imshow('test2', display1)
    cv2.waitKey(10)

    #Fusion===============================================================================================================
    original_image = cv2.imread(image_filename)
    lidar_drivable = np.full((IMG_H, IMG_W), 100)
    camera_drivable = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    fusion_drivable = np.zeros((IMG_H, IMG_W), dtype=float)
    distance_map = np.zeros((IMG_H, IMG_W), dtype=float)
    map_2side_edge = np.zeros((2, 250), dtype=float)

    for i in range(IMG_W):
        lidar_drivable[edge_map_lidar2[0,i]:edge_map_lidar2[1,i], i] = 0

    filtered_lidar = np.zeros((IMG_H, IMG_W), dtype=float)
    filter_1d = np.ones((1, 11)) / (12 * 100)
    subarray = lidar_drivable[600:, :]  ## change
    #filtered_lidar[600:, :] = convolve2d(subarray, filter_1d, mode='same') ##change
    filtered_lidar[600:, :] = subarray/100
    #plt.imshow(filtered_lidar)
    #plt.show()

    #top_right = (rholeftmean - 550 * sl) / cl
    top = (rholeftmean - y_intercept*sl) / cl
    bottom_right = (rholeftmean - 809 * sl) / cl ##change
    #top_left = (rhorightmean - 550 * sr) / cr
    bottom_left = (rhorightmean - 809 * sr) / cr ##change
    #vertices = np.array([[top_left, 550], [top_right, 550], [bottom_right, 799], [bottom_left, 799]], dtype=np.int32)
    vertices = np.array([[top, y_intercept], [bottom_right, 809], [bottom_left, 809]], dtype=np.int32) ##change
    vertices = vertices.reshape((1, -1, 2))
    cv2.fillPoly(camera_drivable, [vertices], 100)

    original_image2 = cv2.imread(image_filename)
    distance_map[:600, :] = 40 ##change
    rows, cols = np.indices(distance_map.shape)
    mask = rows >= 600 ##change
    distance_map[mask] = (IMG_H - rows[mask]) / 1000 * 40
    yes_indices = np.logical_and(lidar_drivable == 100, camera_drivable == 100)  # 1->2
    no_indices = np.logical_and(lidar_drivable != 100, camera_drivable == 100)  # 1->2

    yes_values = np.asarray(filtered_lidar[yes_indices]).reshape(-1, 1)  # 1->2
    fusion_drivable[yes_indices] = fuzzy_fusion_single(yes_values).reshape(-1)
    #input_fuzzy_fusion = np.column_stack((distance_map[no_indices], filtered_lidar[no_indices]))

    drivable_thd = 0.3  # tunable here
    test_result = np.zeros((IMG_H, IMG_W), dtype=np.int64)
    test_result[fusion_drivable > drivable_thd] = 1
    #test_result[(lidar_drivable == 100) & (camera_drivable ==100)] = 1

    #Fusion display before ground truth
    '''
    for i in range(550, 900):
        for j in range(IMG_W):
            if 0.3 < fusion_drivable[i, j] < 0.75:
            #if camera_drivable[i,j] == 100:
                color_depth = [255, 0, 0] #blue
                cv2.circle(original_image2, (j, i), 0, color_depth, thickness=2, lineType=8, shift=0)
            elif fusion_drivable[i, j] >= 0.75:
                color_depth = [0, 0, 255] #red
                cv2.circle(original_image2, (j, i), 0, color_depth, thickness=2, lineType=8, shift=0)

    display3 = original_image2[::2, ::2]
    cv2.imshow('final_fusion', display3)
    cv2.waitKey(10)
    '''
    #==============Test===========================================

    image_element = find_image_by_name(image_elem,sample)
    mask = image_element.find('.//mask')
    label = mask.attrib['label']
    rle = mask.attrib['rle']
    rle_numbers = list(map(int, rle.split(',')))
    gt_top = int(mask.attrib['top'])
    gt_left = int(mask.attrib['left'])
    mask_height = int(mask.attrib['height'])
    mask_width = int(mask.attrib['width'])
    gt_bitmap = np.zeros((IMG_H, IMG_W), dtype=np.int64)

    bitmask = []
    for m in range(0, len(rle_numbers), 2):
        bitmask.extend([0] * rle_numbers[m])
        if m + 1 < len(rle_numbers):
            bitmask.extend([1] * rle_numbers[m + 1])

    bitmask = np.array(bitmask)
    final_mask = bitmask.reshape((mask_height, mask_width))

    gt_bitmap[gt_top:gt_top + mask_height, gt_left:gt_left + mask_width] = final_mask

    for i in range(600, 810): ##change
        for j in range(IMG_W):
            if test_result[i, j] == 1:
                if gt_bitmap[i, j] == 1:
                    color_depth = [255, 0, 0]  # blue correct
                    cv2.circle(original_image2, (j, i), 0, color_depth, thickness=2, lineType=8, shift=0)
                else:
                    color_depth = [0, 255, 0]  # green false alarm
                    cv2.circle(original_image2, (j, i), 0, color_depth, thickness=2, lineType=8, shift=0)
            elif gt_bitmap[i, j] == 1:
                color_depth = [0, 0, 255]  # red missed
                cv2.circle(original_image2, (j, i), 0, color_depth, thickness=2, lineType=8, shift=0)

    display3 = original_image2[::2, ::2]
    cv2.imshow('final_fusion', display3)
    cv2.waitKey(10)
    # plt.imshow(gt_bitmap, cmap='gray')
    # plt.show()
    # plt.imshow(test_result, cmap = 'gray')
    # plt.show()

    iou, f1 = metric_compute(test_result, gt_bitmap)
    print("iou = ", iou)
    print("f1 = ", f1)
    total_iou += iou
    total_f1 += f1

    cv2.waitKey(0)

average_iou = total_iou / frame_cnt
average_f1 = total_f1 / frame_cnt

print('average iou = ', average_iou)
print('average f1 = ', average_f1)

cv2.waitKey(10)