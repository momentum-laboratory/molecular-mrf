import os
import time
import scipy.io as sio

from cest_mrf.metrics.dot_product import dot_prod_matching

from my_funcs.cest_functions import bruker_dataset_creator

phantom_choice = 1  # choose 1-3

# Root stats:
general_fn = os.path.abspath(os.curdir)
# Subject stats:
glu_mouse_fns = [os.path.join(general_fn, 'scans', 'mouse',
                                '20240128_142305_or_dino_mouse2_right_ear_middle_pierce_a_1_1')]  # Mouse
glu_mouse_fn = glu_mouse_fns[phantom_choice-1]
txt_file_name = 'labarchive_notes.txt'
fp_prtcl_name = '51_glu'

glu_mouse_dicom_fn, glu_mouse_mrf_files_fn, bruker_dataset = bruker_dataset_creator(glu_mouse_fn, txt_file_name, fp_prtcl_name)

dict_fn = os.path.join(os.path.join(glu_mouse_mrf_files_fn, 'dict.mat'))
acquired_data_fn = os.path.join(glu_mouse_mrf_files_fn, 'acquired_data.mat')

start = time.perf_counter()
quant_maps = dot_prod_matching(dict_fn=dict_fn, acquired_data_fn=acquired_data_fn)  # I changed it!!!
end = time.perf_counter()
s = (end - start)
print(f"Dot product matching took {s:.03f} s.")

# save acquired data to: root->scans->date->subject->E->mrf_files->quant_maps.mat
quant_maps_fn = os.path.join(glu_mouse_mrf_files_fn, 'quant_maps.mat')
sio.savemat(quant_maps_fn, quant_maps)
print('quant_maps.mat saved')
