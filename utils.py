import cv2
import numpy as np
from skimage import transform as trans

REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
]

DEFAULT_CROP_SIZE = (96, 112)


def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0) make the inner region a square
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    # print('---> default:')
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    if (output_size and
            output_size[0] == tmp_crop_size[0] and
            output_size[1] == tmp_crop_size[1]):
        print('output_size == DEFAULT_CROP_SIZE {}: return default reference points'.format(tmp_crop_size))
        return tmp_5pts

    if (inner_padding_factor == 0 and
            outer_padding == (0, 0)):
        if output_size is None:
            print('No paddings to do: return default reference points')
            return tmp_5pts




    if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0)
            and output_size is None):
        output_size = tmp_crop_size * \
                      (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)
        print('              deduced from paddings, output_size = ', output_size)


    # 1) pad the inner region according inner_padding_factor
    # print('---> STEP1: pad the inner region according inner_padding_factor')
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    # 2) resize the padded inner region
    # print('---> STEP2: resize the padded inner region')
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    # print('              crop_size = ', tmp_crop_size)
    # print('              size_bf_outer_pad = ', size_bf_outer_pad)


    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    # print('              resize scale_factor = ', scale_factor)
    tmp_5pts = tmp_5pts * scale_factor
    #    size_diff = tmp_crop_size * (scale_factor - min(scale_factor))
    #    tmp_5pts = tmp_5pts + size_diff / 2
    tmp_crop_size = size_bf_outer_pad
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    # 3) add outer_padding to make output_size
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size
    # print('---> STEP3: add outer_padding to make output_size')
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)
    #
    # print('===> end get_reference_facial_points\n')

    return reference_5point

def get_affine_transform_matrix(src_pts, dst_pts):
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm


def warp_and_crop_face(src_img,
                       facial_pts,
                       reference_pts =None,
                       crop_size=(96, 112),
                       align_type='smilarity'):
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = (0, 0)
            output_size = crop_size

            reference_pts = get_reference_facial_points(output_size,
                                                        inner_padding_factor,
                                                        outer_padding,
                                                        default_square)

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape


    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape


    if src_pts_shp[0] == 2:
        src_pts = src_pts.T



    if align_type is 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    #        print('cv2.getAffineTransform() returns tfm=\n' + str(tfm))
    elif align_type is 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    #        print('get_affine_transform_matrix() returns tfm=\n' + str(tfm))
    else:
        # tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
        tform = trans.SimilarityTransform()
        tform.estimate(src_pts, ref_pts)
        tfm = tform.params[0:2, :]

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img

def detection_results_show(image_path, faces):

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    for i in range(len(faces[0])):
        cv2.rectangle(image,
                    (int(faces[0][i][0]), int(faces[0][i][1])),
                    (int(faces[0][i][2]), int(faces[0][i][3])),
                    (0,155,255), 2)

        cv2.circle(image,(int(faces[1][0][i]), int(faces[1][5][i])), 2, (0,155,255), 2)
        cv2.circle(image,(int(faces[1][1][i]), int(faces[1][6][i])), 2, (0,155,255), 2)
        cv2.circle(image,(int(faces[1][2][i]), int(faces[1][7][i])), 2, (0,155,255), 2)
        cv2.circle(image,(int(faces[1][3][i]), int(faces[1][8][i])), 2, (0,155,255), 2)
        cv2.circle(image,(int(faces[1][4][i]), int(faces[1][9][i])), 2, (0,155,255), 2)

    cv2.imshow("Input", image)
    cv2.waitKey(0)

def cropped_results_show(image, faces):

    cv2.circle(image,(int(faces[0]), int(faces[5])), 2, (0,155,255), 2)
    cv2.circle(image,(int(faces[1]), int(faces[6])), 2, (0,155,255), 2)
    cv2.circle(image,(int(faces[2]), int(faces[7])), 2, (0,155,255), 2)
    cv2.circle(image,(int(faces[3]), int(faces[8])), 2, (0,155,255), 2)
    cv2.circle(image,(int(faces[4]), int(faces[9])), 2, (0,155,255), 2)

    cv2.imshow("Input", image)
    cv2.waitKey(0)

def cropped_results(image):

    cv2.imshow("Input", image)
    cv2.waitKey(0)

def find_roll(pts):
    return pts[6] - pts[5]

def find_yaw(pts):
    le2n = pts[2] - pts[0]
    re2n = pts[1] - pts[2]
    if le2n < 0 or re2n < 0:
        return 100
    return le2n - re2n

def find_pitch(pts):
    eye_y = (pts[5] + pts[6]) / 2
    mou_y = (pts[8] + pts[9]) / 2
    e2n = eye_y - pts[7]
    n2m = pts[7] - mou_y
    return e2n/n2m

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))