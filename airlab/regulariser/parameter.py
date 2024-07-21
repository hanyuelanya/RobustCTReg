# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import torch as th
import torch.nn.functional as F

import numpy as np


# Regulariser base class (standard from PyTorch)
class _ParameterRegulariser(th.nn.modules.Module):
    def __init__(self, parameter_name, size_average=False, reduce=True):
        super(_ParameterRegulariser, self).__init__()
        self.isWelschEnabled = False
        self._size_average = size_average
        self._reduce = reduce
        self._weight = 1
        self.name = "parent"
        self._parameter_name = parameter_name

    def SetWeight(self, weight):
        print("SetWeight is deprecated. Use set_weight instead.")
        self.set_weight(weight)

    def set_weight(self, weight):
        self._weight = weight

    # conditional return
    def return_loss(self, tensor, welsch_k=1.0):
        if self._size_average and self._reduce:
            return welsch_k * self._weight * tensor.mean()
        if not self._size_average and self._reduce:
            return welsch_k * self._weight * tensor.sum()
        if not self._reduce:
            return welsch_k * self._weight * tensor


"""
    Base class for spatial parameter regulariser
"""


class _SpatialParameterRegulariser(_ParameterRegulariser):
    def __init__(
        self, parameter_name, scaling=[1, 1, 1], size_average=True, reduce=True
    ):
        super(_SpatialParameterRegulariser, self).__init__(
            parameter_name, size_average, reduce
        )

        self._dim = len(scaling)
        self._scaling = scaling
        if len(scaling) == 1:
            self._scaling = np.ones(self._dim) * self._scaling[0]

        self.name = "parent"

    # conditional return
    def return_loss(self, tensor, welsch_k=1.0):
        if self._size_average and self._reduce:
            return welsch_k * self._weight * tensor.mean()
        if not self._size_average and self._reduce:
            return welsch_k * self._weight * tensor.sum()
        if not self._reduce:
            return welsch_k * self._weight * tensor


"""
    Isotropic TV regularisation
"""


class WelschIsoPtvRegulariser(_SpatialParameterRegulariser):
    def __init__(
        self,
        parameter_name,
        vr=math.sqrt(2) / 4,
        scaling=[1],
        size_average=True,
        reduce=True,
    ):
        super(WelschIsoPtvRegulariser, self).__init__(
            parameter_name, scaling, size_average, reduce
        )
        self.isWelschEnabled = True
        self.name = "param_welsch_iso_regular"
        self.parameter_last = -1
        self.init_vr = vr
        self.vr = vr
        self.inverse_2_vr_square = 1 / (2 * vr**2)
        self._dim = 3
        if self._dim == 2:
            self._regulariser = self._regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._regulariser_3d  # 3d regularisation

    def _regulariser_2d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (parameter[:, 1:, 1:] - parameter[:, :-1, 1:]).pow(
                    2
                ) * self._scaling[0]
                dy = (parameter[:, 1:, 1:] - parameter[:, 1:, :-1]).pow(
                    2
                ) * self._scaling[1]
                dx_last = dx.copy().detach()
                dy_last = dy.copy().detach()

                return th.exp(-1 * self.inverse_2_vr_square * (dx_last + dy_last)) * (
                    dx + dy
                )

    def _regulariser_3d(self, parameters):
        for name, parameter in parameters:
            dx = (parameter[:, :, 1:, 1:, 1:] - parameter[:, :, :-1, 1:, 1:]).pow(2) * 1
            dy = (parameter[:, :, 1:, 1:, 1:] - parameter[:, :, 1:, :-1, 1:]).pow(2) * 1
            dz = (parameter[:, :, 1:, 1:, 1:] - parameter[:, :, 1:, 1:, :-1]).pow(2) * 1
            dx_last = dx.clone().detach()
            dy_last = dy.clone().detach()
            dz_last = dz.clone().detach()

            return th.exp(
                -1 * self.inverse_2_vr_square * (dx_last + dy_last + dz_last)
            ) * (dx + dy + dz)

    def getInitV(self):
        return self.init_vr

    def setV(self, vr):
        self.vr = vr
        self.inverse_2_vr_square = 1 / (2 * self.vr**2)

    def forward(self, parameters):

        # set the supgradient to zeros
        value = self._regulariser(parameters)
        mask = value > 0
        # value[mask] = th.sqrt(value[mask])

        return self.return_loss(value, welsch_k=self.inverse_2_vr_square)


class IsotropicTVRegulariser(_SpatialParameterRegulariser):
    def __init__(self, parameter_name, scaling=[1], size_average=True, reduce=True):
        super(IsotropicTVRegulariser, self).__init__(
            parameter_name, scaling, size_average, reduce
        )

        self.name = "param_isoTV"
        self._dim = 3
        if self._dim == 2:
            self._regulariser = self._regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._regulariser_3d  # 3d regularisation

    def _regulariser_2d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (parameter[:, 1:, 1:] - parameter[:, :-1, 1:]).pow(
                    2
                ) * self._scaling[0]
                dy = (parameter[:, 1:, 1:] - parameter[:, 1:, :-1]).pow(
                    2
                ) * self._scaling[1]

                return dx + dy

    def _regulariser_3d(self, parameters):
        for name, parameter in parameters:
            # if self._parameter_name == name:
            dx = (parameter[:, :, 1:, 1:, 1:] - parameter[:, :, :-1, 1:, 1:]).pow(2) * 1
            dy = (parameter[:, :, 1:, 1:, 1:] - parameter[:, :, 1:, :-1, 1:]).pow(2) * 1
            dz = (parameter[:, :, 1:, 1:, 1:] - parameter[:, :, 1:, 1:, :-1]).pow(2) * 1

            return dx + dy + dz

    def forward(self, parameters):

        # set the supgradient to zeros
        value = self._regulariser(parameters)
        mask = value > 0
        value[mask] = th.sqrt(value[mask])

        return self.return_loss(value)

class UnoptimizedMaskWelschIsoPtvRegulariser(_SpatialParameterRegulariser):

    def __init__(
        self,
        parameter_name,
        vr=math.sqrt(2) / 2,
        scaling=[1],
        size_average=True,
        reduce=True,
        mask=None,
    ):
        super(UnoptimizedMaskWelschIsoPtvRegulariser, self).__init__(
            parameter_name, scaling, size_average, reduce
        )

        self.isWelschEnabled = True
        self.name = "param_mask_welsch_ptv_regular"
        self.mask = mask
        self.value_last = 0
        self.parameter_last = -1
        self.init_vr = vr
        self.vr = vr
        self.inverse_2_vr_square = 1 / (2 * self.vr**2)
        self._dim = 3
        if self._dim == 2:
            self._regulariser = self._regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._regulariser_3d  # 3d regularisation

    def _regulariser_2d(self, parameters):
        _, self.parameters_last = self.parameters()
        self.parameters_last = self.parameters_last.clone().detach()
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (parameter[:, 1:, 1:] - parameter[:, :-1, 1:]).pow(
                    2
                ) * self._scaling[0]
                dy = (parameter[:, 1:, 1:] - parameter[:, 1:, :-1]).pow(
                    2
                ) * self._scaling[1]
                dx_last = dx.copy().detach()
                dy_last = dy.copy().detach()

                return th.exp(-1 * self.inverse_2_vr_square * (dx_last + dy_last)) * (
                    dx + dy
                )

    def _regulariser_3d(self, parameters):
        for name, parameter in parameters:
            mask = self.mask.expand(1, 3, *self.mask.shape[2:])
            mask = th.nn.functional.interpolate(mask, parameter.shape[2:])
            # mask[mask>=0] = 1
            self.parameter_last = self.parameter_last * mask
            parameter = parameter * mask
            dx = (parameter[:, :, 1:, 1:, 1:] - parameter[:, :, :-1, 1:, 1:]).pow(2) * 1
            dy = (parameter[:, :, 1:, 1:, 1:] - parameter[:, :, 1:, :-1, 1:]).pow(2) * 1
            dz = (parameter[:, :, 1:, 1:, 1:] - parameter[:, :, 1:, 1:, :-1]).pow(2) * 1


            return 1-th.exp(
               -(dx + dy + dz)*self.inverse_2_vr_square
            ) 

    def getInitV(self):
        return self.init_vr

    def setV(self, vr):
        self.vr = vr
        self.inverse_2_vr_square = 1 / (2 * self.vr**2)

    def forward(self, parameters):

        # set the supgradient to zeros
        value = self._regulariser(parameters)
        mask = value > 0

        return self.return_loss(value, welsch_k=1)

class MaskWelschIsoPtvRegulariser(_SpatialParameterRegulariser):

    def __init__(
        self,
        parameter_name,
        vr=math.sqrt(2) / 2,
        scaling=[1],
        size_average=True,
        reduce=True,
        mask=None,
    ):
        super(MaskWelschIsoPtvRegulariser, self).__init__(
            parameter_name, scaling, size_average, reduce
        )

        self.isWelschEnabled = True
        self.name = "param_mask_welsch_ptv_regular"
        self.mask = mask
        self.value_last = 0
        self.parameter_last = -1
        self.init_vr = vr
        self.vr = vr
        self.inverse_2_vr_square = 1 / (2 * self.vr**2)
        self._dim = 3
        if self._dim == 2:
            self._regulariser = self._regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._regulariser_3d  # 3d regularisation

    def _regulariser_2d(self, parameters):
        _, self.parameters_last = self.parameters()
        self.parameters_last = self.parameters_last.clone().detach()
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (parameter[:, 1:, 1:] - parameter[:, :-1, 1:]).pow(
                    2
                ) * self._scaling[0]
                dy = (parameter[:, 1:, 1:] - parameter[:, 1:, :-1]).pow(
                    2
                ) * self._scaling[1]
                dx_last = dx.copy().detach()
                dy_last = dy.copy().detach()

                return th.exp(-1 * self.inverse_2_vr_square * (dx_last + dy_last)) * (
                    dx + dy
                )

    def _regulariser_3d(self, parameters):
        for name, parameter in parameters:
            mask = self.mask.expand(1, 3, *self.mask.shape[2:])
            mask = th.nn.functional.interpolate(mask, parameter.shape[2:])
            # mask[mask>=0] = 1
            self.parameter_last = self.parameter_last * mask
            parameter = parameter * mask
            dx = (parameter[:, :, 1:, 1:, 1:] - parameter[:, :, :-1, 1:, 1:]).pow(2) * 1
            dy = (parameter[:, :, 1:, 1:, 1:] - parameter[:, :, 1:, :-1, 1:]).pow(2) * 1
            dz = (parameter[:, :, 1:, 1:, 1:] - parameter[:, :, 1:, 1:, :-1]).pow(2) * 1

            dx_last = dx.clone().detach()
            dy_last = dy.clone().detach()
            dz_last = dz.clone().detach()
            return th.exp(
                -1 * self.inverse_2_vr_square * (dx_last + dy_last + dz_last)
            ) * (dx + dy + dz)

    def getInitV(self):
        return self.init_vr

    def setV(self, vr):
        self.vr = vr
        self.inverse_2_vr_square = 1 / (2 * self.vr**2)

    def forward(self, parameters):
        value = self._regulariser(parameters)
        mask = value > 0



        return self.return_loss(value, welsch_k=self.inverse_2_vr_square)


class MaskIsotropicTVRegulariser(_SpatialParameterRegulariser):
    def __init__(
        self, parameter_name, scaling=[1], size_average=True, reduce=True, mask=None
    ):
        super(MaskIsotropicTVRegulariser, self).__init__(
            parameter_name, scaling, size_average, reduce
        )

        self.name = "param_isoTV"
        self._dim = 3
        self.mask = mask
        if self._dim == 2:
            self._regulariser = self._regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._regulariser_3d  # 3d regularisation

    def _regulariser_2d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (parameter[:, 1:, 1:] - parameter[:, :-1, 1:]).pow(
                    2
                ) * self._scaling[0]
                dy = (parameter[:, 1:, 1:] - parameter[:, 1:, :-1]).pow(
                    2
                ) * self._scaling[1]

                return dx + dy

    def _regulariser_3d(self, parameters):
        for name, parameter in parameters:
            mask = self.mask.expand(1, 3, *self.mask.shape[2:])
            mask = th.nn.functional.interpolate(mask, parameter.shape[2:])
            # mask[mask>=0] = 1
            p = parameter * mask
            # if self._parameter_name == name:
            dx = (p[:, :, 1:, 1:, 1:] - p[:, :, :-1, 1:, 1:]).pow(2) * 1
            dy = (p[:, :, 1:, 1:, 1:] - p[:, :, 1:, :-1, 1:]).pow(2) * 1
            dz = (p[:, :, 1:, 1:, 1:] - p[:, :, 1:, 1:, :-1]).pow(2) * 1

            return dx + dy + dz

    def forward(self, parameters):

        # set the supgradient to zeros
        value = self._regulariser(parameters)
        mask = value > 0
        value[mask] = th.sqrt(value[mask])

        return self.return_loss(value)


"""
    TV regularisation 
"""


class TVRegulariser(_SpatialParameterRegulariser):
    def __init__(self, parameter_name, scaling=[1], size_average=True, reduce=True):
        super(TVRegulariser, self).__init__(
            parameter_name, scaling, size_average, reduce
        )

        self.name = "param_TV"
        self._dim == 3
        self._regulariser = self._regulariser_3d
        if self._dim == 2:
            self._regulariser = self._regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._regulariser_3d  # 3d regularisation

    def _regulariser_2d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (
                    th.abs(parameter[:, 1:, 1:] - parameter[:, :-1, 1:])
                    * self._pixel_spacing[0]
                )
                dy = (
                    th.abs(parameter[:, 1:, 1:] - parameter[:, 1:, :-1])
                    * self._pixel_spacing[1]
                )

                return dx + dy

    def _regulariser_3d(self, parameters):
        for name, parameter in parameters:
            # if self._parameter_name in name:

            dx = th.abs(parameter[0, :, 1:, 1:, 1:] - parameter[0, :, :-1, 1:, 1:]) * 1
            dy = th.abs(parameter[0, :, 1:, 1:, 1:] - parameter[0, :, 1:, :-1, 1:]) * 1
            dz = th.abs(parameter[0, :, 1:, 1:, 1:] - parameter[0, :, 1:, 1:, :-1]) * 1
            return dx + dy + dz

    def forward(self, parameters):
        return self.return_loss(self._regulariser(parameters))


class MaskTVRegulariser(_SpatialParameterRegulariser):
    def __init__(
        self, parameter_name, scaling=[1], size_average=True, reduce=True, mask=None
    ):
        super(MaskTVRegulariser, self).__init__(
            parameter_name, scaling, size_average, reduce
        )

        self.name = "param_TV"
        self._dim == 3
        self._regulariser = self._regulariser_3d
        self.mask = mask
        if self._dim == 2:
            self._regulariser = self._regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._regulariser_3d  # 3d regularisation

    def _regulariser_2d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (
                    th.abs(parameter[:, 1:, 1:] - parameter[:, :-1, 1:])
                    * self._pixel_spacing[0]
                )
                dy = (
                    th.abs(parameter[:, 1:, 1:] - parameter[:, 1:, :-1])
                    * self._pixel_spacing[1]
                )

                return dx + dy

    def _regulariser_3d(self, parameters):
        for name, parameter in parameters:
            # if self._parameter_name in name:
            mask = self.mask.expand(1, 3, *self.mask.shape[2:])
            mask = th.nn.functional.interpolate(mask, parameter.shape[2:])
            # mask[mask>=0] = 1
            p = parameter * mask
            dx = th.abs(p[0, :, 1:, 1:, 1:] - p[0, :, :-1, 1:, 1:]) * 1
            dy = th.abs(p[0, :, 1:, 1:, 1:] - p[0, :, 1:, :-1, 1:]) * 1
            dz = th.abs(p[0, :, 1:, 1:, 1:] - p[0, :, 1:, 1:, :-1]) * 1
            return dx + dy + dz

    def forward(self, parameters):
        return self.return_loss(self._regulariser(parameters))


"""
    Diffusion regularisation 
"""


class DiffusionRegulariser(_SpatialParameterRegulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(DiffusionRegulariser, self).__init__(
            pixel_spacing, [1], size_average, reduce
        )

        self.name = "param diff"
        self._dim = 3
        if self._dim == 2:
            self._regulariser = self._regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._regulariser_3d  # 3d regularisation

    def _regulariser_2d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (parameter[:, 1:, 1:] - parameter[:, :-1, 1:]).pow(
                    2
                ) * self._pixel_spacing[0]
                dy = (parameter[:, 1:, 1:] - parameter[:, 1:, :-1]).pow(
                    2
                ) * self._pixel_spacing[1]

                return dx + dy

    def _regulariser_3d(self, parameters):
        for name, parameter in parameters:
            # if self._parameter_name in name:
            dx = (parameter[0, :, 1:, 1:, 1:] - parameter[0, :, :-1, 1:, 1:]).pow(2) * 1
            dy = (parameter[0, :, 1:, 1:, 1:] - parameter[0, :, 1:, :-1, 1:]).pow(2) * 1
            dz = (parameter[0, :, 1:, 1:, 1:] - parameter[0, :, 1:, 1:, :-1]).pow(2) * 1

            return dx + dy + dz

    def forward(self, parameters):
        return self.return_loss(self._regulariser(parameters))


class MaskDiffusionRegulariser(_SpatialParameterRegulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True, mask=None):
        super(MaskDiffusionRegulariser, self).__init__(
            pixel_spacing, [1], size_average, reduce
        )

        self.name = "param diff"
        self._dim = 3
        self.mask = mask
        if self._dim == 2:
            self._regulariser = self._regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._regulariser_3d  # 3d regularisation

    def _regulariser_2d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (parameter[:, 1:, 1:] - parameter[:, :-1, 1:]).pow(
                    2
                ) * self._pixel_spacing[0]
                dy = (parameter[:, 1:, 1:] - parameter[:, 1:, :-1]).pow(
                    2
                ) * self._pixel_spacing[1]

                return dx + dy

    def _regulariser_3d(self, parameters):
        for name, parameter in parameters:
            mask = self.mask.expand(1, 3, *self.mask.shape[2:])
            mask = th.nn.functional.interpolate(mask, parameter.shape[2:])
            # mask[mask>=0]=1
            # if self._parameter_name in name:
            p = parameter * mask
            dx = (p[0, :, 1:, 1:, 1:] - p[0, :, :-1, 1:, 1:]).pow(2) * 1
            dy = (p[0, :, 1:, 1:, 1:] - p[0, :, 1:, :-1, 1:]).pow(2) * 1
            dz = (p[0, :, 1:, 1:, 1:] - p[0, :, 1:, 1:, :-1]).pow(2) * 1

            return dx + dy + dz

    def forward(self, parameters):
        return self.return_loss(self._regulariser(parameters))


class JacobianRegulariser(_SpatialParameterRegulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(JacobianRegulariser, self).__init__(
            pixel_spacing, [1], size_average, reduce
        )

        self.name = "Jacobian"
        self._dim = 3
        if self._dim == 2:
            self._regulariser = self._joc_regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._joc_regulariser_3d  # 3d regularisation

    def Get_Ja(self, parameters):
        for name, parameter in parameters:
            dx = parameter[0, :, 1:, 1:, 1:] - parameter[0, :, :-1, 1:, 1:]
            dy = parameter[0, :, 1:, 1:, 1:] - parameter[0, :, 1:, :-1, 1:]
            dz = parameter[0, :, 1:, 1:, 1:] - parameter[0, :, 1:, 1:, :-1]

            D1 = (dx[0, ...] + 1) * (
                (dy[1, ...] + 1) * (dz[2, ...] + 1) - dz[1, ...] * dy[2, ...]
            )
            D2 = (dx[1, ...]) * (
                dy[0, ...] * (dz[2, ...] + 1) - dy[2, ...] * dz[0, ...]
            )
            D3 = (dx[2, ...]) * (
                dy[0, ...] * dz[1, ...] - (dy[1, ...] + 1) * dz[0, ...]
            )

            D = D1 - D2 + D3
            a = D[D < 0].sum()
            return D

    def _joc_regulariser_2d(self, displacement):
        pass

    def _joc_regulariser_3d(self, parameters):
        absj = self.Get_Ja(parameters)
        Neg_Jac = 0.5 * (th.abs(absj) - absj)
        return Neg_Jac
        # print("j: ",Neg_Jac.mean())

    def forward(self, parameters):
        return self.return_loss(self._regulariser(parameters))


"""
    Sparsity regularisation 
"""


class MaskWelschSparsityRegulariser(_ParameterRegulariser):
    def __init__(
        self, parameter_name, vr=0.5, size_average=True, reduce=True, mask=None
    ):
        super(MaskWelschSparsityRegulariser, self).__init__(
            parameter_name, size_average, reduce
        )

        self.name = "param_L1"
        self.mask = mask

        self.init_vr = vr
        self.vr = vr
        self.inverse_2_vr_square = 1 / (2 * self.vr**2)

        self.value_last = -1

    def getInitV(self):
        return self.init_vr

    def setV(self, vr):
        self.vr = vr
        self.inverse_2_vr_square = 1 / (2 * self.vr**2)

    def forward(self, parameters):
        # [1,3,5,5,5]
        # [5,5,5]
        # [1,1,27,19,24]
        for name, parameter in parameters:
            mask = self.mask.expand(1, 3, *self.mask.shape[2:])
            mask = th.nn.functional.interpolate(mask, parameter.shape[2:])
            # mask[mask>=0]=1
            value = th.abs(parameter * mask)
            self.value_last = value.clone().detach()
            value = th.exp(-1 * self.inverse_2_vr_square * self.value_last) * value
            return self.return_loss(value, self.inverse_2_vr_square)


class MaskSparsityRegulariser(_ParameterRegulariser):
    def __init__(self, parameter_name, size_average=True, reduce=True, mask=None):
        super(MaskSparsityRegulariser, self).__init__(
            parameter_name, size_average, reduce
        )

        self.name = "param_L1"
        self.mask = mask

    def forward(self, parameters):
        # [1,3,5,5,5]
        # [5,5,5]
        # [1,1,27,19,24]
        for name, parameter in parameters:
            mask = self.mask.expand(1, 3, *self.mask.shape[2:])
            mask = th.nn.functional.interpolate(mask, parameter.shape[2:])
            # mask[mask>=0]=1
            return self.return_loss(th.abs(parameter * mask))


class SparsityRegulariser(_ParameterRegulariser):
    def __init__(self, parameter_name, size_average=True, reduce=True):
        super(SparsityRegulariser, self).__init__(parameter_name, size_average, reduce)

        self.name = "param_L1"

    def forward(self, parameters):
        # [1,3,5,5,5]
        # [5,5,5]
        # [1,1,27,19,24]
        for name, parameter in parameters:
            return self.return_loss(th.abs(parameter))
