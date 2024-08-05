# requires subject_dicts['save_name'] = save_name

# This scan has no 107a!
subject_dicts_jan_24 = [
    {'scan_name': '24_01_28_glu_mouse_37deg',
     'sub_name': '1_or_dino_mouse2_right_ear_middle_pierce',
     'month': 'jan',
     'center': [0, 0],
     'voxel_origin': [0, 0],
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 0,
     'resratio': 2,
     'roi_center': [0, 0],
     'temp': 37,
     'z_b1s': [0.7, 2],
     'z_b1s_names': ['0p7uT', '2uT']
     },
    ]

# These mice are young and have tumors
subject_dicts_april = [
    {'scan_name': '24_04_04_glu_tumor_mouse_37deg',
     'sub_name': '1_mouse1_left',
     'month': 'apr',
     'center': [39, 15],
     'voxel_origin': [28, 13],
     't_shift': [1, 8],  # up, right
     'highres_img_idx': 2,
     'resratio': 4,
     'roi_center': [0, 0],
     'temp': 37,
     'z_b1s': [0.7, 1.5],
     'z_b1s_names': ['0p7uT', '1p5uT']
     },
    {'scan_name': '24_04_04_glu_tumor_mouse_37deg',
     'sub_name': '2_mouse2_right',
     'month': 'apr',
     'center': [25, 14],
     'voxel_origin': [16, 11],
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 1,
     'resratio': 4,
     'roi_center': [0, 0],
     'temp': 37,
     'z_b1s': [0.7, 1.5],
     'z_b1s_names': ['0p7uT', '1p5uT']
     },
        {'scan_name': '24_04_04_glu_tumor_mouse_37deg',
     'sub_name': '3_mouse3_two_ears',
     'month': 'apr',
     'center': [30, 19],
     'voxel_origin': [21, 14],
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 1,
     'resratio': 4,
     'roi_center': [0, 0],
     'temp': 37,
     'z_b1s': [0.7, 1.5],
     'z_b1s_names': ['0p7uT', '1p5uT']
     },
 ]


# These mice are old
subject_dicts_june_old = [
        {'scan_name': '24_06_03_glu_old_mouse_37deg',
     'sub_name': '1_mouse_old_tail_broken',
     'month': 'june',
     'center': [39, 15],
     'voxel_origin': [28, 13],
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 1,
     'resratio': 2,
     'roi_center': [20, 45],
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2uT', '2p5uT']
     },
        {'scan_name': '24_06_03_glu_old_mouse_37deg',
     'sub_name': '2_mouse_old_female_3bends',
     'month': 'june',
     'center': [39, 15],
     'voxel_origin': [28, 13],
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 1,
     'resratio': 2,
     'roi_center': [27, 41],
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2uT', '2p5uT']
     },
        {'scan_name': '24_06_03_glu_old_mouse_37deg',
     'sub_name': '3_mouse_old_male_2bends',
     'month': 'june',
     'center': [39, 15],
     'voxel_origin': [28, 13],
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 1,
     'resratio': 2,
     'roi_center': [26, 46],
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2uT', '2p5uT']
     },
        {'scan_name': '24_06_03_glu_old_mouse_37deg',
     'sub_name': '4_mouse_old_male_1band',
     'month': 'june',
     'center': [39, 15],
     'voxel_origin': [28, 13],
     't_shift': [-1, 0],  # up, right
     'highres_img_idx': 0,
     'resratio': 2,
     'roi_center': [30, 41],
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2uT', '2p5uT']
     },
 ]


# These mice are young and have tumors
subject_dicts_june_tumor_10 = [
        {'scan_name': '24_06_10_glu_tumor_mouse_37deg',
     'sub_name': '2_mouse_no_tumor_day_10',
     'month': 'june',
     'center': [39, 15],
     'voxel_origin': [28, 13],
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 0,
     'resratio': 2,
     'roi_center': [40, 20],
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2p5uT']  # no reverse (first instance)
     },
        {'scan_name': '24_06_10_glu_tumor_mouse_37deg',
     'sub_name': '5_mouse_no_tumor_day_10',
     'month': 'june',
     'center': [39, 15],
     'voxel_origin': [28, 13],
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 1,
     'resratio': 2,
     'roi_center': [31, 40],
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2p5uT']  # slice matching issue
     },
        {'scan_name': '24_06_10_glu_tumor_mouse_37deg',
     'sub_name': '8_mouse_2R_tumor_day_10',
     'month': 'june',
     'center': [39, 15],
     'voxel_origin': [28, 13],
     't_shift': [-1, 0],  # up, right
     'highres_img_idx': 1,
     'resratio': 2,
     'roi_center': [26, 43],
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2p5uT']
     },
 ]

subject_dicts_june_tumor_14 = [
        {'scan_name': '24_06_13_glu_tumor_mouse_37deg',
     'sub_name': '2_mouse_no_tumor_day_14',
     'month': 'june',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 0,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2p5uT']
     },
        {'scan_name': '24_06_13_glu_tumor_mouse_37deg',
     'sub_name': '5_mouse_no_tumor_day_14',
     'month': 'june',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [-1, -1],  # up, right
     'highres_img_idx': 1,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2p5uT']  # slice matching issue
     },
        {'scan_name': '24_06_13_glu_tumor_mouse_37deg',
     'sub_name': '8_mouse_2R_tumor_day_14',
     'month': 'june',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [-1, 0],  # up, right
     'highres_img_idx': 1,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2p5uT']
     },
 ]

subject_dicts_june_tumor_17 = [
        {'scan_name': '24_06_16_glu_tumor_mouse_37deg',
     'sub_name': '2_mouse_no_tumor_day_17',
     'month': 'june',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 0,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2p5uT']
     },
        {'scan_name': '24_06_16_glu_tumor_mouse_37deg',
     'sub_name': '5_mouse_no_tumor_day_17',
     'month': 'june',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [-1, -1],  # up, right
     'highres_img_idx': 1,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2p5uT']  # slice matching issue
     },
        {'scan_name': '24_06_16_glu_tumor_mouse_37deg',
     'sub_name': '8_mouse_2R_tumor_day_17',
     'month': 'june',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [-1, 0],  # up, right
     'highres_img_idx': 1,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [0.7, 1.5, 2.5],
     'z_b1s_names': ['0p7uT', '1p5uT', '2p5uT']
     },
 ]


subject_dicts_july_ped_tumor_8 = [
        {'scan_name': '24_07_08_pediatric_tumor_mouse_37deg',
     'sub_name': '1L',
     'month': 'july',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [0, 0],  # down, left
     'highres_img_idx': 0,
     'mask_slice': 0,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [1.5],
     'z_b1s_names': ['1p5uT']
     },
        {'scan_name': '24_07_08_pediatric_tumor_mouse_37deg',
     'sub_name': '1R',
     'month': 'july',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 0,
     'mask_slice': 0,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [1.5],
     'z_b1s_names': ['1p5uT']
     },
    {'scan_name': '24_07_08_pediatric_tumor_mouse_37deg',
     'sub_name': '1R1L',
     'month': 'july',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 1,
     'mask_slice': 0,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [1.5],
     'z_b1s_names': ['1p5uT']
     },
        {'scan_name': '24_07_08_pediatric_tumor_mouse_37deg',
     'sub_name': '2L',
     'month': 'july',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 3,
     'mask_slice': 2,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [1.5],
     'z_b1s_names': ['1p5uT']
     },
 ]

subject_dicts_july_ped_tumor_15 = [
        {'scan_name': '24_07_15_pediatric_tumor_mouse_37deg',
     'sub_name': '1L',
     'month': 'july',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [0, 0],  # down, left
     'highres_img_idx': 2,
     'mask_slice': 1,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [1.5],
     'z_b1s_names': ['1p5uT']
     },
        {'scan_name': '24_07_15_pediatric_tumor_mouse_37deg',
     'sub_name': '1R',
     'month': 'july',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 2,
     'mask_slice': 1,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [1.5],
     'z_b1s_names': ['1p5uT']
     },
    {'scan_name': '24_07_15_pediatric_tumor_mouse_37deg',
     'sub_name': '1R1L',
     'month': 'july',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 2,
     'mask_slice': 1,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [1.5],
     'z_b1s_names': ['1p5uT']
     },
        {'scan_name': '24_07_15_pediatric_tumor_mouse_37deg',
     'sub_name': '2L',
     'month': 'july',
     'center': [0, 0],  # do I ever use this?
     'voxel_origin': [0, 0],  # do I ever use this?
     't_shift': [0, 0],  # up, right
     'highres_img_idx': 1,
     'mask_slice': 0,
     'resratio': 2,
     'temp': 37,
     'z_b1s': [1.5],
     'z_b1s_names': ['1p5uT']
     },
 ]