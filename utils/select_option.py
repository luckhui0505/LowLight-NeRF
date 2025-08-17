# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from typing import *

import gin

from src.data.litdata import (
    LitDataBlender,
    LitDataBlenderMultiScale,
    LitDataLF,
    LitDataLLFF,
    LitDataLLFF_Pair,
    LitDataNeRF360V2,
    LitDataRefNeRFReal,
    LitDataShinyBlender,
    LitDataTnT,
    LitDataSingle
)
from src.model.dvgo.model import LitDVGO
from src.model.mipnerf.model import LitMipNeRF
from src.model.nerf.model import LitNeRF
from src.model.nerfpp.model import LitNeRFPP
from src.model.plenoxel.model import LitPlenoxel
from src.model.refnerf.model import LitRefNeRF
from src.model.aleth_nerf.model import LitAleth_NeRF
from src.model.aleth_nerf_exp.model import LitAleth_NeRF_Exp


def select_model(
    model_name: str, eta:float =gin.REQUIRED, con:float =gin.REQUIRED
):

    if model_name == "nerf":
        return LitNeRF()
    elif model_name == "mipnerf":
        return LitMipNeRF()
    elif model_name == "plenoxel":
        return LitPlenoxel()
    elif model_name == "nerfpp":
        return LitNeRFPP()
    elif model_name == "dvgo":
        return LitDVGO()
    elif model_name == "refnerf":
        return LitRefNeRF()
    elif model_name == 'aleth_nerf':
        return LitAleth_NeRF(eta=eta, con=con)
    elif model_name == 'aleth_nerf_exp':
        return LitAleth_NeRF_Exp(eta=eta, con=con)

    else:
        raise f"Unknown model named {model_name}"


def select_dataset(
    dataset_name: str,
    datadir: str,
    scene_name: str,
):
    if dataset_name == "blender":
        data_fun = LitDataBlender
    elif dataset_name == "blender_multiscale":
        data_fun = LitDataBlenderMultiScale
    elif dataset_name == "llff":
        data_fun = LitDataLLFF
    elif dataset_name == 'llff_pair':
        data_fun = LitDataLLFF_Pair
    elif dataset_name == "tanks_and_temples":
        data_fun = LitDataTnT
    elif dataset_name == "lf":
        data_fun = LitDataLF
    elif dataset_name == "nerf_360_v2":
        data_fun = LitDataNeRF360V2
    elif dataset_name == "shiny_blender":
        data_fun = LitDataShinyBlender
    elif dataset_name == "refnerf_real":
        data_fun = LitDataRefNeRFReal
    elif dataset_name == 'single_image':
        data_fun = LitDataSingle
    
    

    return data_fun(
        datadir=datadir,
        scene_name=scene_name,
    )


def select_callback(model_name):

    callbacks = []

    if model_name == "plenoxel":
        import src.model.plenoxel.model as model

        callbacks += [model.ResampleCallBack()]

    if model_name == "dvgo":
        import src.model.dvgo.model as model

        callbacks += [
            model.Coarse2Fine(),
            model.ProgressiveScaling(),
            model.UpdateOccupancyMask(),
        ]

    return callbacks
