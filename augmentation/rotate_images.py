import cv2
from math import cos, sin, acos, asin
import numpy as np
import scipy.io
from scipy.io import savemat
from scipy import ndimage
import os

# images to be rotated (it's assumed that accompaying .mat files are located in the same directory)
INPUT_DIR_PATH = 'LFPW/'
OUTPUT_DIR_PATH = 'LFPW_rotated_scaled/'
ANGLES = list(range(-20, 30, 10))
ANGLES.remove(0)


def modify_landmarks(data_path, landmarks_path):
    dir_list = os.listdir(landmarks_path)
    for dir in dir_list:
        dir_path = os.path.join(landmarks_path, dir)
        file_list = os.listdir(dir_path)
        for file in file_list:
            file_path = os.path.join(dir_path, file)
            pts_mat = loadmat(file_path)
            mat_path = os.path.join(data_path, dir + "/" + file[:-8] + ".mat")
            flip_mat_path = os.path.join(data_path, dir + "_Flip/" + file[:-8] + ".mat")
            mat = loadmat(mat_path)
            flip_mat = loadmat(flip_mat_path)
            for i, (x, y) in enumerate(pts_mat["pts_2d"]):
                mat['pt2d'][0, i], mat['pt2d'][1, i] = x, y
                flip_mat['pt2d'][0, i], flip_mat['pt2d'][1, i] = 450 - x, y
            savemat(mat_path, mat)
            savemat(flip_mat_path, flip_mat)
            print("set {} landmarks over".format(file))
            

def euler2matrix(pitch, yaw, roll, mode="rotation"):
    """
    transfer Euler angle 2 rotation matrix
    :param pitch: angle of rotation with X-axis
    :param yaw: angle of rotation with Z-axis
    :param roll: angle of rotation with Y-axis
    :param mode: rotation or orientation
    :return: orientation matrix
    """
    assert mode in ["rotation", "orientation"], "mode should be rotation and orientation"
    assert np.min(np.abs([pitch, yaw, roll])) <= np.pi, "absolution of pitch,yaw and roll should less than PI"

    if mode == "rotation":
        # rotation matrix of pitch
        mx = np.mat([[1, 0, 0],
                     [0, cos(pitch), -sin(pitch)],
                     [0, sin(pitch), cos(pitch)]])
        # rotation matrix of yaw
        my = np.mat([[cos(yaw), 0, sin(yaw)],
                     [0, 1, 0],
                     [-sin(yaw), 0, cos(yaw)]])
        # rotation matrix of roll
        mz = np.mat([[cos(roll), -sin(roll), 0],
                     [sin(roll), cos(roll), 0],
                     [0, 0, 1]])
        matrix = mz * my * mx
    else:
        # orientation matrix of pitch
        mx = np.mat([[1, 0, 0],
                     [0, cos(pitch), sin(pitch)],
                     [0, -sin(pitch), cos(pitch)]])
        # orientation matrix of yaw
        my = np.mat([[cos(yaw), 0, -sin(yaw)],
                     [0, 1, 0],
                     [sin(yaw), 0, cos(yaw)]])
        # orientation matrix of roll
        mz = np.mat([[cos(roll), sin(roll), 0],
                     [-sin(roll), cos(roll), 0],
                     [0, 0, 1]])
        matrix = mx * my * mz

    return matrix


def matrix_l2r(matrix, axis='x'):
    """
    transfer matrix from left coordinate to right or reverse(right to left)
    :param matrix: original matrix
    :param axis: axis of the reverse one
    :return:matrix
    """
    assert axis in ['x', 'y', 'z'], "param of 'axis' should be 'x','y' or 'z'"
    if axis == 'x':
        s = np.mat([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif axis == 'y':
        s = np.mat([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        s = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    return s * matrix * s


def matrix2euler(matrix):
    """
    transfer rotation matrix to Euler-Angle(Rotation Rz Ry Rx)
    :param matrix:np.mat(3,3)
    :return:
    """
    if abs(matrix[2, 0]) != 1:
        yaw = asin(-matrix[2, 0])
        pitch = atan2(matrix[2, 1], matrix[2, 2])
        roll = atan2(matrix[1, 0], matrix[0, 0])
    elif matrix[2, 0] == -1:
        yaw = np.pi / 2
        roll = np.pi / 4
        pitch = atan2(matrix[0, 1], matrix[0, 2]) + roll
    else:
        yaw = -np.pi / 2
        roll = np.pi / 4
        pitch = atan2(-matrix[0, 1], -matrix[0, 2]) + roll

    return np.hstack([pitch, yaw, roll])



def save_ypr(yaw, pitch, roll, pt2d, mat_file_name):
    yaw = yaw * np.pi / 180
    pitch = pitch * np.pi / 180
    roll = roll * np.pi / 180

    mat = dict()
    mat['Pose_Para'] = [[pitch, yaw, roll]]
    mat['pt2d'] = pt2d

    savemat(mat_file_name, mat)


# transform yaw, pitch and roll
def rotate_pyr(pitch, yaw, roll, angle):
    # get rotation matrix(Rz->Ry->Rx) from origin Euler-Angle
    base_matrix = euler2matrix(pitch, yaw, roll)

    # rotate Z-axis with angle again.so get the last rotation matrix
    # Notice: The return matrix is under the rotation condition(not orientation)
    angle = angle * np.pi / 180
    last_matrix = base_matrix * np.mat([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])

    return matrix2euler(last_matrix)


# transform pt2d
def rotate_pt2d(pt2d, org_img, rotated_img, angle):
    pt2d_new = np.copy(pt2d)
    angle = angle * np.pi / 180
    matrix = np.mat([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])
    matrix = matrix_l2r(matrix, 'y')  # transfer the matrix of right-hand coordinate  to left one
    org_ch, org_cw, _ = np.array(org_img.shape) / 2
    rot_ch, rot_cw, _ = np.array(rotated_img.shape) / 2

    for pt in range(pt2d_new.shape[1]):
        # non scaled new coordinates
        # rotation the original point[x,y,0] with matrix
        x, y, _ = (np.mat([pt2d_new[0, pt] - org_cw, pt2d_new[1, pt] - org_ch, 0]) * matrix).__array__()[0]
        # scale coordinates
        x += rot_cw
        y += rot_ch
        pt2d_new[0, pt], pt2d_new[1, pt] = x, y

    # get rotated pt2d
    return pt2d_new


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_DIR_PATH):
        os.mkdir(OUTPUT_DIR_PATH)

    all_files = os.listdir(INPUT_DIR_PATH)
    counter = 0
    for mat_f in all_files:
        if mat_f.endswith('.mat') and np.random.random_sample() < 0.05:

            base_name = mat_f.split('.')[0]

            mat = scipy.io.loadmat(os.path.join(INPUT_DIR_PATH, mat_f))
            pitch = mat['Pose_Para'][0][0]
            yaw = mat['Pose_Para'][0][1]
            roll = mat['Pose_Para'][0][2]

            img = cv2.imread(os.path.join(INPUT_DIR_PATH, base_name + ".jpg"))

            for angle in ANGLES:
                # rotate image,  and landmarks
                img_rotated = ndimage.rotate(img, -angle)
                r_pitch, r_yaw, r_roll = rotate_pyr(pitch, yaw, roll, angle)

                pt2d = mat['pt2d']
                pt2d_rotated = rotate_pt2d(pt2d, img, img_rotated, angle)

                # save image and .mat file
                cv2.imwrite(os.path.join(OUTPUT_DIR_PATH, base_name + "_rotated_%s.jpg" % str(angle)), img_rotated)
                save_ypr(r_yaw, r_pitch, r_roll, pt2d_rotated,
                         os.path.join(OUTPUT_DIR_PATH, base_name + "_rotated_%s.mat" % str(angle)))

            counter += 1
            if counter % 100 == 0:
                print(str(counter) + " images processed so far")
