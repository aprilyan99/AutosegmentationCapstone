import os 
import numpy as np
import shutil

def generate_patient_list(data_path, input_tag):
    folders = sorted(os.listdir(data_path + "/"))[2:]
    file_names = [path for path in sorted(os.listdir(data_path + input_tag + "/" + folders[0])) if path[0].isalpha()]
    return file_names

def create_dir(root_dir, task_name, i):
    path = root_dir + task_name + file_names[i][-10:-7] + '/'
    os.makedirs(path)
    isExist = os.path.exists(path) 
    if not isExist:
        try:
            os.makedirs(path, exist_ok = True)
            print("Directory '%s' created successfully" % path)
        except OSError as error:
            print("Directory '%s' can not be created" % path)

def move_file_rename(root_dir_ori, root_dir_tar, type_name, task_name, tar_name, i):
    ori_path = root_dir_ori + type_name + task_name + file_names[i][-10:-7] + '.nii.gz'
    tar_path = root_dir_tar + task_name + file_names[i][-10:-7] + '/' + tar_name
    isExist = os.path.exists(tar_path)
    if not isExist:
        try:
            shutil.move(ori_path, tar_path)
            print("Directory '%s' moved successfully" % tar_path)
        except OSError as error:
            print("Directory '%s' can not be moved" % tar_path)

def move_folder(root_dir, task_name, type_name, i):
    ori_path = root_dir + task_name + file_names[i][-10:-7]
    tar_path = root_dir + type_name + task_name + file_names[i][-10:-7]
    isExist = os.path.exists(tar_path)
    if not isExist:
        try:
            shutil.move(ori_path, tar_path)
            print("Directory '%s' moved successfully" % tar_path)
        except OSError as error:
            print("Directory '%s' can not be moved" % tar_path)
