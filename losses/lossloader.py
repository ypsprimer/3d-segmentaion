from .jupiter_loss import *
from .loss_skeleton import *

class LossLoader:
    def __init__(self):
        pass

    @staticmethod
    def load(lossname, config):

        ignore_index = config['prepare']['label_para']['ignore_index']
        if isinstance(lossname, list):
            object = []
            for loss in lossname:
                object.append(globals()[loss](ignore_index=ignore_index))
        else:
            object = globals()[lossname](ignore_index=ignore_index)

        return object


if __name__ == "__main__":
    loss = LossLoader.load("DiceLoss")
    print(loss)
