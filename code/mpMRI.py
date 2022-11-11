## save data in 4 channels
# root_dir = '/Users/aprilyan/Documents/capstone/MSD/Patients/'
folders = ['Test','Train','Val']
for folder in folders:
    task_dir = sorted(os.listdir(root_dir + folder))[:]
    for i in task_dir:
        img_path = root_dir + folder + '/' + i + '/image.nii.gz'
        if os.path.exists(img_path):
            image = nib.load(img_path)
            img = image.get_fdata()
            for j in range(img.shape[-1]):
                channel = img[...,j]
                new_affine = image.affine
                new_header = image.header
                new_img = nib.Nifti1Image(channel, new_affine, new_header)
                nib.save(new_img, root_dir + folder + '/' + str(i) + '/image' + str(j) + '.nii.gz')
            os.remove(img_path)
