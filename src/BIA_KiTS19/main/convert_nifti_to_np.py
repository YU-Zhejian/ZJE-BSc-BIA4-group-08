from typing import Tuple

import SimpleITK as sitk

from BIA_KiTS19.helper.sitk_helper import resample_spacing

if __name__ == '__main__':
    image1 = sitk.ReadImage("/media/yuzj/BUP/kits19/data/case_00155/imaging.nii.gz")
    image2 = sitk.ReadImage("/media/yuzj/BUP/kits19/data/case_00155/segmentation.nii.gz")
    image1_rs = resample_spacing(image1)
    image2_rs = resample_spacing(image2)
    image1_rs.GetSize(), image2_rs.GetSize()
    image1_rs.GetOrigin(), image2_rs.GetOrigin()
    image2_rs_arr = sitk.GetArrayViewFromImage(image2_rs)
    image2_rs_arr_f = image2_rs_arr.flatten()
