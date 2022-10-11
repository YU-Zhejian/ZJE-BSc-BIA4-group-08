from typing import Tuple

import SimpleITK as sitk


def resample_spacing(
        sitk_image: sitk.Image,
        newspace: Tuple[int, int, int]=(1, 1, 1)
) -> sitk.Image:
    img_size = sitk_image.GetSize()
    img_spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    direction = sitk_image.GetDirection()
    new_size = tuple(map(
        lambda i: int(img_size[i] * img_spacing[i] / newspace[i]),
        range(3)
    ))
    sitk_image = sitk.Resample(
        image1=sitk_image,
        size=new_size,
        transform=sitk.Euler3DTransform(),
        interpolator=sitk.sitkLinear,
        outputOrigin=origin,
        outputSpacing=newspace,
        outputDirection=direction
    )
    return sitk_image

