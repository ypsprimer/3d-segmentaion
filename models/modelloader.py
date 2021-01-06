from .unets_liver_thick import unet_a_liver_thick_sync_plus
from .unets_liver_thin import unet_a_liver_thin_sync_plus, unet_a_liver_thin_sync
# from .shared_decoder import *
# from .multi_class_shared_decoder import multi_class_unet_shared_encoders_medium
from .multi_class_ThiCK_shared_decoder import multi_class_ThiCK_unet_shared_encoders_medium
# from .multi_class_SectionSeg import multi_class_SectionSeg_unet_shared_encoders_medium
# from .multi_class_SectionSeg_AddVesselMask import *
# from .multi_class_VesselSeg import multi_class_VesselSeg_unet_regular

class ModelLoader:
    def __init__(self):
        pass

    @staticmethod
    def load(modelname, *args, **kwargs):
        object = globals()[modelname]
        return object(*args, **kwargs)


if __name__ == "__main__":
    net = ModelLoader.load("unet_c3_d_sync")
    print(net.cuda())
