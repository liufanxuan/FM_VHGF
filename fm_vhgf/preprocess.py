"""Crop co-registered PET(SUV), CT(HU) and tumour-mask NIfTI inputs.

PET must already be in SUV units. This module does not infer dose, patient
weight, decay correction, DICOM scaling, or a registration transform.
"""

import argparse
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output


def crop_at_center(array, center, size=96, fill=0):
    shape = np.full(3, size, dtype=int) if np.isscalar(size) else np.asarray(size, dtype=int)
    start = np.floor(np.asarray(center) - (shape - 1) / 2).astype(int)
    end = start + shape
    source_start, source_end = np.maximum(start, 0), np.minimum(end, array.shape)
    output = np.full(tuple(shape), fill, dtype=np.float32)
    if np.all(source_end > source_start):
        destination_start = source_start - start
        destination_end = destination_start + source_end - source_start
        output[tuple(slice(a, b) for a, b in zip(destination_start, destination_end))] = (
            array[tuple(slice(a, b) for a, b in zip(source_start, source_end))]
        )
    return output


def prepare_case(pet_path, ct_path, mask_path, output, label=None, crop_size=96):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pet, ct, mask = (nib.load(str(path)) for path in (pet_path, ct_path, mask_path))
    for image in (pet, ct, mask):
        if len(image.shape) != 3 or not np.isfinite(image.affine).all():
            raise ValueError("Expected 3D NIfTI images with finite spatial transforms.")
    # All three images are placed on one CT-derived 1 mm physical grid.
    ct_grid = resample_to_output(ct, voxel_sizes=(1, 1, 1), order=1, cval=-1000)
    target = (ct_grid.shape, ct_grid.affine)
    pet_array = resample_from_to(pet, target, order=1, cval=0).get_fdata(dtype=np.float32)
    ct_array = ct_grid.get_fdata(dtype=np.float32)
    mask_array = resample_from_to(mask, target, order=0, cval=0).get_fdata()
    if not all(np.isfinite(array).all() for array in (pet_array, ct_array, mask_array)):
        raise ValueError("Input images contain NaN or infinity.")
    indices = np.argwhere(mask_array > 0)
    if not len(indices):
        raise ValueError("The tumour annotation is empty after spatial resampling.")
    # Bounding box of all annotated intraprostatic regions, not a component centroid.
    center = (indices.min(0) + indices.max(0)) / 2
    ct_array = np.clip((ct_array - (50 - 350 / 2)) / 350, 0, 1)
    pet_crop = crop_at_center(pet_array, center, crop_size)
    ct_crop = crop_at_center(ct_array, center, crop_size)
    if label is not None and label not in (0, 1):
        raise ValueError("Provide a binary postoperative reference label coded 0/1.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5:
        # NIfTI spatial axes are XYZ; the network consumes CDHW = CZYX.
        h5.create_dataset("pet", data=pet_crop.transpose(2, 1, 0)[None], compression="gzip")
        h5.create_dataset("ct", data=ct_crop.transpose(2, 1, 0)[None], compression="gzip")
        if label is not None:
            h5.create_dataset("label", data=np.int64(label))
        h5.attrs["spacing_mm"] = (1.0, 1.0, 1.0)
        h5.attrs["ct_window_width"] = 350
        h5.attrs["ct_window_level"] = 50
        h5.attrs["pet_units"] = "SUV; provided by the input image"
        h5.attrs["crop_center_xyz"] = center
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pet-suv", "ct-hu", "mask", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--label", type=int, choices=[0, 1])
    args = parser.parse_args(argv)
    return prepare_case(args.pet_suv, args.ct_hu, args.mask, args.output, args.label)


if __name__ == "__main__":
    main()
