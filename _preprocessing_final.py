import os
import multiprocessing

from joblib import Parallel, delayed
from _preprocessing_functions import *

#path to nifti directory
nifti_dir = '/home/neural_network_code/Data/Patients/' #local 
vols_to_process = ['MPRAGE.nii.gz']
rois_to_process = ['ROI.nii.gz']

#path to SLICER3D (used for reorienting, registering, resampling volumes)
slicer_dir = '/home/shared_software/Slicer-4.10.2-linux-amd64/Slicer'
#path to ROBEX (used for skull stripping volumes)
robex_dir = '/home/shared_software/ROBEX/runROBEX.sh' #comment out

# parameter to run orientation module
orientation = 'RAI'
# parameters to run resampling module
spacing = '1,1,1'
interp_type_vol = 'bspline'
interp_type_roi = 'nearestNeighbor'
# parameter to run bias correction module
n4_iterations = [50,50,30,20]
# parameter to run skull strip module (need to find T1C volume out of list of all vols_to_process)
volume_to_skullstrip = vols_to_process[0][:-7] + '_'

#####################################################################################
#run preprocessing over all patients
patients = nested_folder_filepaths(nifti_dir, vols_to_process)
patients.sort()

def all_preprocessing(patient):
    if len(rois_to_process) == 0 or (len(rois_to_process) > 0 and not os.path.exists(os.path.join(nifti_dir+patient, rois_to_process[0][:-7] + '_RAI_RESAMPLED_BINARY-label.nii.gz'))):
        # 1) reorient volumes and rois
        reoriented_volumes = reorient_volume(nifti_dir, patient, vols_to_process, orientation, slicer_dir)
        if len(rois_to_process) > 0:
            reoriented_rois = reorient_volume(nifti_dir, patient, rois_to_process, orientation, slicer_dir)
        # 2) resample to isotropic resolution
        resampled_volumes = resample_volume(nifti_dir, patient, reoriented_volumes, spacing, interp_type_vol, slicer_dir)
        if len(rois_to_process) > 0:
            resampled_rois = resample_volume(nifti_dir, patient, reoriented_rois, spacing, interp_type_roi, slicer_dir)
        # 3) n4 bias correction
        # get initial skull mask to use in N4
        temp_skull_stripped_volume = skull_strip(nifti_dir, patient, volume_to_skullstrip, [resampled_volumes[0]], robex_dir)
        temp_skull_mask = get_non_zero_mask(nifti_dir, patient, temp_skull_stripped_volume)
        # sometimes there is a mismatch between the mask and the volume, so we will copy the affine/header from the volume of interest
        temp_skull_mask = replace_affine_header(nifti_dir, patient, temp_skull_mask, resampled_volumes[0])
        bias_corrected_volumes = n4_bias_correction(nifti_dir, patient, resampled_volumes, n4_iterations, mask_image=temp_skull_mask[0])
        # remove temporary skull masks
        os.remove(os.path.join(nifti_dir + patient, temp_skull_stripped_volume[0]))
        os.remove(os.path.join(nifti_dir + patient, temp_skull_mask[0]))
        # 4) skull stripping of N4 bias corrected images
        skull_stripped_volumes = skull_strip(nifti_dir, patient, volume_to_skullstrip, bias_corrected_volumes, robex_dir)
        # 5) patient level normalization of skull-stripped volumes using only non-zero elements
        normalized_SS_volumes = normalize_volume(nifti_dir, patient, skull_stripped_volumes)
        # 6) patient level normalization of non-skull-stripped volumes using only voxels from inside skull region
        normalized_non_SS_volumes = normalize_volume(nifti_dir, patient, bias_corrected_volumes, reference_volume=skull_stripped_volumes[0])
        # 7) binarize ROI
        if len(rois_to_process) > 0:
            binarized_roi = binarize_segmentation(nifti_dir, patient, resampled_rois)

num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(all_preprocessing)(patient) for patient in patients)
