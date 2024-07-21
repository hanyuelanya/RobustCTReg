import os
import numpy as np
import SimpleITK as sitk
from skimage import measure
from lungmask import mask

root_path = r"E:\Desktop\DIR-Lab\4DCT\mha"
case_folders = os.listdir(root_path)

def clamp(input_path):

    image = sitk.ReadImage(input_path)

    clamped = sitk.Clamp(image, lowerBound=0, upperBound=2000)

    output_path = input_path.replace(".mha", "_clamped.mha")
    sitk.WriteImage(clamped, output_path)

def get_bone_mask(image,save_path, threshold=1200):
    # Read the image from the specified path
    vol = sitk.GetArrayFromImage(image)
    # Threshold the image
    vol[vol < threshold] = 0
    vol[vol >= threshold] = 1

    # Perform binary morphological closing
    vol = sitk.GetImageFromArray(vol)
    bm = sitk.BinaryMorphologicalClosingImageFilter()
    bm.SetKernelType(sitk.sitkBall)
    bm.SetKernelRadius(4)
    bm.SetForegroundValue(1)
    vol = bm.Execute(vol)
    vol = sitk.GetArrayFromImage(vol)

    # Remove small connected components
    label = measure.label(vol, connectivity=2)
    props = measure.regionprops(label)
    for ia in range(len(props)):
        if props[ia].area <= 10:
            label[label == props[ia].label] = 0
    label[label > 1] = 1
    label = label.astype(np.int16)
    # Save the resulting image
    sitk.WriteImage(sitk.GetImageFromArray(label), save_path)


def segment_lung(image,save_path):
    segmentation = mask.apply(image,batch_size=15)  # default model is U-net(R231)
    segmentation=segmentation.astype(np.int16)
    sitk.WriteImage(sitk.GetImageFromArray(segmentation), save_path)


def process_points(path):
    with open(path, 'r') as f_in:
        lines = f_in.readlines()
        buffer = []
        for line in lines:
            nums = line.strip().split('\t')
            nums[0] = str(int(nums[0]) - 1)
            nums[1] = str(int(nums[1]) - 1)
            buffer.append('\t'.join(nums) + '\n')

        if len(buffer) > 0:
            with open(path.replace(".pts", "_processed.pts"), 'a') as f_out:
                f_out.writelines(buffer)

def main():


    path = r"E:\Desktop\Structure_Aware_Registration-master\Structure_Aware_Registration-master\DataSet\Dirlab4DCT\Case1Pack\Resample\C1T50_r.mha"
    image = sitk.ReadImage(path)
    get_bone_mask(image,path.replace(".mha", "_boneMask.mha"))
    segment_lung(sitk.Subtract(image, 1024),path.replace(".mha", "_lungMask.mha"))

if __name__ == '__main__':
    # saveMidData()
    # main()
    process_points(r"E:\Desktop\Structure_Aware_Registration-master\Structure_Aware_Registration-master\DataSet\Dirlab4DCT\Case1Pack\Pts\C1T50_300.pts")




