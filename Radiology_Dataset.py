import numpy as np
import SimpleITK as sitk
import torch
import os
from glob import glob
from pathlib import Path
from torch.utils.data import Dataset


def downsamplePatient(patient_CT, height, width, depth):
    original_CT = sitk.ReadImage(patient_CT, sitk.sitkInt32)
    # original_CT = patient_CT
    dimension = original_CT.GetDimension()
    reference_physical_size = np.zeros(original_CT.GetDimension())
    reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                  zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]

    reference_origin = original_CT.GetOrigin()
    reference_direction = original_CT.GetDirection()

    # reference_size = [round(sz / resize_factor) for sz in original_CT.GetSize()]
    reference_size = [height, width, depth]
    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(original_CT.GetDirection())

    transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)

    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    # centered_transform.AddTransform(centering_transform)
    final_transform = sitk.CompositeTransform([centered_transform, centering_transform])
    # sitk.Show(sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0))

    return sitk.Resample(original_CT, reference_image, final_transform, sitk.sitkLinear, 0.0)


class RadiologyDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # image = nib.load(self.image_paths[idx])
        # img = image.get_fdata()
        # img = (img-img.min())/img.max()
        # print("Reading ", self.image_paths[idx])
        subject_name = Path(self.image_paths[idx]).stem
        image = downsamplePatient(self.image_paths[idx], 128, 128, 128)
        img = sitk.GetArrayFromImage(image)
        img_tensor = torch.unsqueeze(torch.Tensor(img), 0)
        lbl = self.labels[idx]
        return img_tensor, lbl, subject_name
