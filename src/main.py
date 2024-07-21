import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from mymedicallib import utils as mut
import time
import gc
import torch as th
import numpy as np
import SimpleITK as sitk
from dirlab.dirlabhelper import Dataloader
import torch.nn.functional as F
import copy
import airlab as al
import json
import pandas as pd
from copy import deepcopy

# from apex import amp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
dataset = 'copd'
# dataset = 'dirlab'

def create_mask_pyramid(mask, image_pyramid):
    mask_pyramid = []
    image_dim = 3
    for img in image_pyramid:
        shape = img.image.shape[2:]
        mask_sample = F.interpolate(mask.image, size=shape, mode="trilinear")
        mask_sample[mask_sample >= 0.5] = 1
        mask_sample[mask_sample < 0.5] = 0
        mask_size = img.size[-image_dim:]
        mask_spacing = img.spacing
        mask_origin = img.origin
        mask_pyramid.append(al.Image(mask_sample, mask_size, mask_spacing, mask_origin))
    return mask_pyramid


def gap_overlap(vol1, vol2, label=1):
    vol1[vol1 > 0] = 1
    vol1[vol1 < 1] = 0
    vol2[vol2 > 0] = 1
    vol2[vol2 < 1] = 0
    vol1l = vol1 == label
    vol2l = vol2 == label
    s = np.sum(np.logical_and(vol1l, vol2l))
    vol1l = ~vol1l
    vol2l = ~vol2l
    o = np.sum(np.logical_and(vol1l, vol2l))
    return s, o


using_landmarks = True


class Registration:
    def __init__(self, configpath, dataconfigpath, appRoot):
        self.config = None
        with open(configpath) as f:
            self.config = json.load(f)
        self.dataconfigpath = dataconfigpath
        self.dtype = th.float32
        self.device = "cuda"
        self.appRoot = appRoot

    def loaddata(self, case="1"):
        """
        load image
        """
        print("loading images")
        # loader = Dataloader(self.config['dataset']['dataset'], self.dataconfigpath,True,True)
        loader = Dataloader(dataset, self.dataconfigpath, True, True)

        global_fix = al.Image(
            loader.GetVolumeData(case=case, T="00" if dataset=='dirlab' else 'e0', voltype="sitk"),
            self.dtype,
            self.device,
        )
        global_mov = al.Image(
            loader.GetVolumeData(case=case, T="50" if dataset=='dirlab' else 'i0', voltype="sitk"),
            self.dtype,
            self.device,
        )
        global_fix, global_mov = al.utils.normalize_images(global_fix, global_mov)

        lung_fix = loader.GetLungMask(case=case, T="00" if dataset=='dirlab' else 'e0', voltype="sitk")
        lung_mov = loader.GetLungMask(case=case, T="50" if dataset=='dirlab' else 'i0', voltype="sitk")

        bone_fix = loader.GetBoneMask(case=case,T="00" if dataset=='dirlab' else 'e0', voltype="sitk")
        bone_mov = loader.GetBoneMask(case=case, T="50" if dataset=='dirlab' else 'i0', voltype="sitk")

        body_fix = mut.getbody(lung_fix)
        body_mov = mut.getbody(lung_mov)

        surface_fix, spine_fix = mut.getlungsurface(lung_fix, bone_fix)
        other_fix = mut.getother(surface_fix, spine_fix)

        lung_fix = al.Image(lung_fix, self.dtype, self.device)
        lung_fix.image = lung_fix.image * global_fix.image
        lung_mov = al.Image(lung_mov, self.dtype, self.device)
        lung_mov.image = lung_mov.image * global_mov.image

        bone_fix = al.Image(bone_fix, self.dtype, self.device)
        bone_fix.image = bone_fix.image * global_fix.image
        bone_mov = al.Image(bone_mov, self.dtype, self.device)
        bone_mov.image = bone_mov.image * global_mov.image

        body_fix = al.Image(body_fix, self.dtype, self.device)
        body_fix.image = body_fix.image * global_fix.image
        body_mov = al.Image(body_mov, self.dtype, self.device)
        body_mov.image = body_mov.image * global_mov.image

        surface_fix = al.Image(surface_fix, self.dtype, self.device)
        spine_fix = al.Image(spine_fix, self.dtype, self.device)
        other_fix = al.Image(other_fix, self.dtype, self.device)

        points_fix = loader.GetPtsData(
            case=case, T="00" if dataset=='dirlab' else 'e0', spacing=loader.GetVolSpacing(case)
        )
        points_mov = loader.GetPtsData(
            case=case, T="50" if dataset=='dirlab' else 'i0', spacing=loader.GetVolSpacing(case)
        )
        points = {"fix": points_fix, "mov": points_mov}
        pyramid = self.config["hyperparameter"]["pyramid"]
        im_pyramid = {}
        mask_pyramid = {}
        if self.config["method"]["is_global_loss"]:
            im_pyramid["global"] = {
                "mov": al.create_image_pyramid(global_mov, pyramid),
                "fix": al.create_image_pyramid(global_fix, pyramid),
            }
        if self.config["method"]["is_bone_loss"]:
            im_pyramid["bone"] = {
                "mov": al.create_image_pyramid(bone_mov, pyramid),
                "fix": al.create_image_pyramid(bone_fix, pyramid),
            }
        if self.config["method"]["is_lung_loss"]:
            im_pyramid["lung"] = {
                "mov": al.create_image_pyramid(lung_mov, pyramid),
                "fix": al.create_image_pyramid(lung_fix, pyramid),
            }
        if self.config["method"]["is_body_loss"]:
            im_pyramid["body"] = {
                "mov": al.create_image_pyramid(body_mov, pyramid),
                "fix": al.create_image_pyramid(body_fix, pyramid),
            }

        if self.config["method"]["is_surface_reg"]:
            mask_pyramid["surface"] = create_mask_pyramid(
                surface_fix, im_pyramid["lung"]["fix"]
            )
        if self.config["method"]["is_spine_reg"]:
            mask_pyramid["spine"] = create_mask_pyramid(
                spine_fix, im_pyramid["lung"]["fix"]
            )
        if self.config["method"]["is_other_reg"]:
            mask_pyramid["other"] = create_mask_pyramid(
                other_fix, im_pyramid["lung"]["fix"]
            )
        size_pyramid = []
        for k in im_pyramid:
            for vol in im_pyramid[k]["mov"]:
                size_pyramid.append(vol.size)
            if len(size_pyramid) > 0:
                break
        return im_pyramid, mask_pyramid, points, size_pyramid

    #

    def setLoss(self, im_pyramid, mask_pyramid, level):
        """
        set loss function
        """
        loss = []

        if self.config["method"]["is_global_loss"]:
            global_fix_level = im_pyramid["global"]["fix"][level]
            global_mov_level = im_pyramid["global"]["mov"][level]

            if self.config["method"]["im_sim"] == "ncc":
                mloss = al.loss.pairwise.NCC(global_fix_level, global_mov_level)

            elif self.config["method"]["im_sim"] == "ngf":
                mloss = al.loss.pairwise.NGF(global_fix_level, global_mov_level)
            elif self.config["method"]["im_sim"] == "welsch_ngf":
                mloss = al.loss.pairwise.WelschNGF(global_fix_level, global_mov_level)
            elif self.config["method"]["im_sim"] == "lcc":
                mloss = al.loss.pairwise.LCC(global_fix_level, global_mov_level)
            mloss.set_loss_weight(1)
            loss.append(mloss)


        if self.config["method"]["is_lung_loss"]:
            lung_fix_level = im_pyramid["lung"]["fix"][level]
            lung_mov_level = im_pyramid["lung"]["mov"][level]
            mloss = al.loss.pairwise.WelschNGF(lung_fix_level, lung_mov_level)
            mloss.set_loss_weight(self.config["hyperparameter"]["lung_w"])
            loss.append(mloss)


        if self.config["method"]["is_bone_loss"]:
            bone_fix_level = im_pyramid["bone"]["fix"][level]
            bone_mov_level = im_pyramid["bone"]["mov"][level]

            mloss = al.loss.pairwise.NCC(bone_fix_level, bone_mov_level)
            mloss.set_loss_weight(self.config["hyperparameter"]["bone_w"])
            loss.append(mloss)


        if self.config["method"]["is_body_loss"]:
            body_fix_level = im_pyramid["body"]["fix"][level]
            body_mov_level = im_pyramid["body"]["mov"][level]

            mloss = al.loss.pairwise.WelschNGF(body_fix_level, body_mov_level)
            mloss.set_loss_weight(self.config["hyperparameter"]["body_w"])
            loss.append(mloss)
        return loss

    def setRegularization(self, im_pyramid, mask_pyramid, level):
        """
        set regularzation
        """
        reg = []
        key = None
        for k in im_pyramid:
            key = k
            break
        global_fix_level = im_pyramid[key]["fix"][level]

        if self.config["method"]["is_global_reg"]:

            if self.config["method"]["is_pTV"]:
                TVreg = al.regulariser.parameter.IsotropicTVRegulariser(
                    global_fix_level.spacing
                )
                TVreg.set_weight(self.config["hyperparameter"]["global_r_w"])
                reg.append(TVreg)

            elif self.config["method"]["is_pSM"]:
                diffreg = al.regulariser.parameter.DiffusionRegulariser(
                    global_fix_level.spacing
                )
                diffreg.set_weight(self.config["hyperparameter"]["global_r_w"])
                reg.append(diffreg)



        if self.config["method"]["is_other_reg"]:
            mmask = mask_pyramid["other"][level].image
            other_reg = al.regulariser.parameter.MaskWelschIsoPtvRegulariser(
                global_fix_level.spacing, mask=mmask
            )
            
            other_reg.set_weight(self.config["hyperparameter"]["other_r_w"])
            reg.append(other_reg)


        if self.config["method"]["is_spine_reg"]:
            mmask = mask_pyramid["spine"][level].image

            spine_reg = al.regulariser.parameter.MaskSparsityRegulariser(
                global_fix_level.spacing, mask=mmask
            )
            spine_reg.set_weight(self.config["hyperparameter"]["spine_r_w"])
            reg.append(spine_reg)

        if self.config["method"]["is_surface_reg"]:
            mmask = mask_pyramid["surface"][level].image

            surface_reg = al.regulariser.parameter.MaskWelschIsoPtvRegulariser(
                global_fix_level.spacing, mask=mmask
            )
            surface_reg.set_weight(self.config["hyperparameter"]["surface_r_w"])
            reg.append(surface_reg)

        return reg

    def start(self, case):
        """
        start registration
        """

        im_pyramid, mask_pyramid, points, size_pyramid = self.loaddata(case=case)
        points_fix, points_mov = points["fix"], points["mov"]
        constant_flow = None
        nlevel = self.config["hyperparameter"]["nlevel"]
        startTime = time.time()
        initV=[]
        for level in range(nlevel):
            loss = self.setLoss(im_pyramid, mask_pyramid, level)
            reg = self.setRegularization(im_pyramid, mask_pyramid, level)
            index=0
            if level != 0:
                for term in loss:
                    if term.isWelschEnabled:
                        term.setV(max(initV[index]/ 1.5**level, 0.3))
                        index+=1
                for term in reg:
                    if term.isWelschEnabled:
                        term.setV(term.getInitV() / 2**level)

            constant_flow, current_displacement = self.register(
                im_pyramid, level, constant_flow, loss, reg, size_pyramid
            )
            
            if level == 0:
                for term in loss:
                    if term.isWelschEnabled:
                        initV.append(term.vd)

            
            # generate SimpleITK displacement field and calculate TRE
            tmp_displacement = al.transformation.utils.upsample_displacement(
                current_displacement.clone().to(device=self.device),
                size_pyramid[-1],
                interpolation="linear",
            )

            tmp_displacement = (
                al.transformation.utils.unit_displacement_to_dispalcement(
                    tmp_displacement
                )
            )  # unit measures to image domain measures
            tmp_displacement = al.Displacement(
                tmp_displacement, size_pyramid[-1], [1, 1, 1], [0, 0, 0]
            )

            # in order to not invert the displacement field, the fixed points are transformed to match the moving points

            if using_landmarks:
                print(
                    "TRE on that level: "
                    + str(
                        al.Points.TRE(
                            points_mov,
                            al.Points.transform(points_fix, tmp_displacement),
                        )
                    )
                )

            del tmp_displacement
            del loss
            del reg
            if level != 0:
                for key in im_pyramid:
                    im_pyramid[key]["mov"][level - 1] = 0
                    im_pyramid[key]["fix"][level - 1] = 0
                for key in mask_pyramid:
                    mask_pyramid[key][level - 1] = 0
                gc.collect()
                th.cuda.empty_cache()

        unit_displacement = current_displacement

        endTime = time.time()
        return unit_displacement, startTime, endTime

    def SetHyperparameter(self, key, value):
        self.config["hyperparameter"][key] = value

    def register(self, im_pyramid, level, constant_flow, loss, reg, size_pyramid):
        """
        registration one level
        """
        print("---- Level " + str(level) + " ----")
        registration = al.PairwiseRegistration(verbose=False)
        transformation = al.transformation.pairwise.BsplineTransformation(
            size_pyramid[level],
            sigma=self.config["hyperparameter"]["sigma"],
            order=3,
            dtype=self.dtype,
            device=self.device,
            diffeomorphic=False,
        )

        if level > 0:
            constant_flow = al.transformation.utils.upsample_displacement(
                constant_flow, size_pyramid[level], interpolation="linear"
            )
            transformation.set_constant_flow(constant_flow)
        registration.set_transformation(transformation)

        registration.set_image_loss(loss)
        registration.set_regulariser_parameter(reg)

        optimizer = th.optim.Adam(
            transformation.parameters(),
            lr=self.config["hyperparameter"]["step_size"][level],
            weight_decay=0.0001,
            #   weight_decay=0.01,
            amsgrad=True,
        )

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(
            self.config["hyperparameter"]["niter"][level]
        )
        registration.start()

        # store current flow field
        constant_flow = transformation.get_flow()
        current_displacement = transformation.get_displacement()
        # del optimizer
        return constant_flow, current_displacement

    def evaluation(self, displacement, startTime, endTime, case):
        """
        evaluate result
        Case, TRE, Bone_DICE, Lung_DICE, Jacobian, Time
        """
        loader = Dataloader(
            self.config["dataset"]["dataset"], self.dataconfigpath, True, True
        )
        points_fix = loader.GetPtsData(
            case=case, T="00" if dataset=='dirlab' else 'e0', spacing=loader.GetVolSpacing(case)
        )
        points_mov = loader.GetPtsData(
            case=case,T="50" if dataset=='dirlab' else 'i0', spacing=loader.GetVolSpacing(case)
        )
        bone_fix = loader.GetBoneMask(case=case, T="00" if dataset=='dirlab' else 'e0', voltype="sitk")
        bone_mov = loader.GetBoneMask(case=case, T="50" if dataset=='dirlab' else 'i0', voltype="sitk")
        lung_fix = loader.GetLungMask(case=case, T="00" if dataset=='dirlab' else 'e0', voltype="sitk")
        lung_mov = loader.GetLungMask(case=case,  T="50" if dataset=='dirlab' else 'i0', voltype="sitk")

        Cost_Time = self.GetTime(startTime, endTime)
        ph_displacement = al.transformation.utils.unit_displacement_to_dispalcement(
            displacement
        )  # unit measures to image domain measures
        ph_displacement = al.Displacement(
            displacement, bone_fix.GetSize(), [1, 1, 1], [0, 0, 0]
        )
        import os
        filename = 'd.npy'
        counter = 1
        new_filename = filename
        while os.path.exists(r"../results"+os.sep+new_filename):
            base_name, extension = os.path.splitext(filename)
            new_filename = f"{base_name}_{counter}{extension}"
            counter += 1
#         np.save("../results"+os.sep+new_filename, np.asarray(deepcopy(ph_displacement).numpy(), dtype=np.float32))
        Init_TRE, TRE,std = self.GetTRE(points_fix, points_mov, ph_displacement)

        print("TRE: ", Init_TRE, TRE)
        print("TREStd:",std)
        print("Time: ", Cost_Time)

        evaluation = [
            case,
            Init_TRE,
            TRE,
            std,
            Cost_Time,
        ]

        return evaluation

    def GetTRE(self, points_fix, points_mov, displacement):
        fixed_points_transformed = al.Points.transform(points_fix, displacement)
        TRE,std = al.Points.TRE(points_mov, fixed_points_transformed, sp=None)
        InitTRE,_ = al.Points.TRE(points_mov, points_fix)
        return round(InitTRE, 6), round(TRE, 6), round(std, 6)


    def GetTime(self, startTime, endTime):
        return round(endTime - startTime, 6)



    def startAll(self):
        result = []
        for i in range(1,11):
            case = str(i)
            unit_displacement, startTime, endTime = self.start(case)
            evaluation = self.evaluation(
                unit_displacement.to(th.float64), startTime, endTime, case
            )
            result.append(evaluation)
            del unit_displacement


    def startAllAnalysisSigmas(self):
        randid = str(random.randint(0, 99999))
        result = []
        sigmas = [
            [12, 12, 12],
            [11, 11, 11],
            [10, 10, 10],
            [9, 9, 9],
            [8, 8, 8],
            [7, 7, 7],
            [6, 6, 6],
        ]
        for sigma in sigmas:
            self.config["hyperparameter"]["sigma"] = sigma
            s = [str(i) for i in sigma]
            postfix = "sigmas_" + randid + "_".join(s)
            for i in range(1, 11):
                case = str(i)
                unit_displacement, startTime, endTime = self.start(case)
                evalueation = self.evaluation(
                    unit_displacement.to(th.float64), startTime, endTime, case
                )
                result.append(evalueation)
                del unit_displacement

#             self.logger(result, postfix)
            result = []



def main():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    configpath = script_dir + "/config_xin.json"
    dataconfigpath = script_dir + "/dataset.json"
    appRoot = script_dir
    registration = Registration(configpath, dataconfigpath, appRoot)

    registration.startAll()
    print("=================================================================")


if __name__ == "__main__":
    main()
