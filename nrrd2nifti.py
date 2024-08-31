import SimpleITK as sitk

img = sitk.ReadImage("segm.nrrd")
sitk.WriteImage(img, "segm.nii.gz")
