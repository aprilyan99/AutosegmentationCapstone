%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
import os
import nibabel as nib
import SimpleITK as sitk
from nipype.interfaces.ants import N4BiasFieldCorrection

import os
import multiprocessing
from subprocess import call
from joblib import Parallel, delayed

# data_path = "Task04_Hippocampus/"  # "Task01_BrainTumour/"
# folders = sorted(os.listdir(data_path))[2:]
# file_names = [path for path in sorted(os.listdir(data_path + folders[0])) if path[0].isalpha()]

# n4_iterations = [50,50,30,20]
# spacing = '1,1,1'
# interp_type = 'nearestNeighbor'
# work_dir = '/Users/aprilyan/Documents/capstone/MSD/'
# interp_type = 'nearestNeighbor'
# slicer_dir = '/Users/aprilyan/Applications/Slicer-4.11.20210226-macosx-amd64/Slicer'


def generate_filepath(work_dir, data_path, folder_tag, subfolder, input_filename):
    """
    subfolder = '/imagesTr', '/labelsTr'
    """
    input_filepath = work_dir + data_path[:-1] + subfolder +'/' + input_filename
    output_filepath = work_dir + data_path[:-1] + folder_tag + subfolder + '/' + input_filename
    
    return input_filepath, output_filepath

def resample_vol_slicer(work_dir, data_path, voxel_spacing, slicer_dir, interp_type,
                        subfolder, input_filename, folder_tag=""):
    input_filepath, output_filepath = generate_filepath(work_dir, data_path, folder_tag, subfolder, input_filename)
    module_name = 'ResampleScalarVolume'
    resample_scalar_volume_command = [slicer_dir,'--launch', module_name, '"' + input_filepath + '" "' + output_filepath + '"', 
                                      '-i', interp_type, '-s', voxel_spacing]
    call(' '.join(resample_scalar_volume_command), shell=True)
    return output_filepath

def resampling_vol(work_dir, data_path, voxel_spacing, subfolder, input_filename, folder_tag=""):
    """
    input_filepath: one for each iteration
    voxel_spacing = (1,1,1,1)
    """
    input_filepath, output_filepath = generate_filepath(work_dir, data_path, folder_tag, subfolder, input_filename)
    nib_vol = nib.load(input_filepath)
    affine = nib_vol.get_affine()
    header = nib_vol.get_header()
    vol = nib_vol.get_data()
    header.set_zooms(voxel_spacing)
    nib_vol_resampled = nib.Nifti1Image(vol, affine, header=header)
    nib.save(nib_vol_resampled, output_filepath)
    
    return output_filepath

def n4_bias_correction(n4_iterations, work_dir, data_path, input_filename, subfolder, folder_tag='_process', image_dim=4):
    """
    folder_tag='_process/imagesTr'
    """
    input_filepath, output_filepath = generate_filepath(work_dir, data_path, folder_tag, subfolder, input_filename)
    n4 = N4BiasFieldCorrection(output_image = output_filepath)
    n4.inputs.dimension = image_dim
    n4.inputs.input_image = input_filepath 
    n4.inputs.n_iterations = n4_iterations 
    n4.run()
    
    return output_filepath

def vol_normalization(work_dir, data_path, input_filename, subfolder,
                     only_nonzero=False, reference_volume = None, skull_mask_volume=None,
                      normalization_params=np.array([]), folder_tag='_normal'):
    """
    folder_tag='_normal/imagesTr'
    """
    input_filepath, output_filepath = generate_filepath(work_dir, data_path, folder_tag, subfolder, input_filename)
    nib_vol = nib.load(input_filepath)
    affine = nib_vol.get_affine()
    header = nib_vol.get_header()
    vol = nib_vol.get_fdata()

    if len(normalization_params) > 0 and len(normalization_params.shape) == 1:
        normalization_params = np.tile(normalization_params, (len(vols_to_process), 1))
    if reference_volume != None:
        reference_vol = nib.load(os.path.join(nifti_dir + patient, reference_volume)).get_data()
        skull_mask = (reference_vol != 0).astype(np.int)
    if skull_mask_volume != None:
        skull_mask_vol = nib.load(os.path.join(nifti_dir + patient, skull_mask_volume)).get_data()
        skull_mask = (skull_mask_vol != 0).astype(np.int)

    #Normalize only non-zero intensity values (if flag set to true)
    if only_nonzero == True and reference_volume == None and skull_mask_volume == None:
        idx_nz = np.nonzero(vol)
    elif only_nonzero == True and (reference_volume != None or skull_mask_volume != None):
        idx_nz = np.nonzero(skull_mask)
    else:
        idx_nz = np.where(vol)
    
    if len(normalization_params) == 0:
        mean, std = np.mean(vol[idx_nz]), np.std(vol[idx_nz])
    else:
        mean, std = normalization_params[i, :]
    vol_norm = np.copy(vol)
    
    if reference_volume == None:
        vol_norm[idx_nz] = (vol_norm[idx_nz] - mean) / std
    else:
        vol_norm = (vol_norm - mean) / std
    nib_vol_norm = nib.Nifti1Image(vol_norm, affine, header=header)
    nib.save(nib_vol_norm, output_filepath)
    
    return output_filepath

def all_preprocessing(patient):
#     resampled_img = resample_vol_slicer(work_dir, data_path, spacing, slicer_dir, interp_type,
#                         subfolder='/imagesTr', input_filename=patient, folder_tag="")
#     resampled_roi = resample_vol_slicer(work_dir, data_path, spacing, slicer_dir, interp_type,
#                         subfolder='/labelsTr', input_filename=patient, folder_tag="")
#     resampled_img = resample_vol_slicer(work_dir, data_path, voxel_spacing=(1.0, 1.0, 1.0, 1.0), 
#                                    input_filename=patient, subfolder='/imagesTr', folder_tag="")
#     resampled_roi = resampling_vol(work_dir, data_path, voxel_spacing=(1.0, 1.0, 1.0), 
#                                    input_filename=patient, subfolder='/labelsTr', folder_tag="")
    process_img = n4_bias_correction(n4_iterations, work_dir, data_path, 
                                     input_filename=patient, subfolder='/imagesTr', folder_tag='_process',
                                    image_dim=3)
    normalize_img = vol_normalization(work_dir, data_path, 
                                      input_filename=patient, subfolder='/imagesTr',
                                      only_nonzero=False, reference_volume=None, skull_mask_volume=None,
                                      normalization_params=np.array([]), folder_tag='_normal')
    return print('Done')

# num_cores = multiprocessing.cpu_count()
# Parallel(n_jobs=num_cores)(delayed(all_preprocessing)(patient) for patient in file_names)




