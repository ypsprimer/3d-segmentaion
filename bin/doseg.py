import os, time, sys

work_path = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(work_path)
import numpy as np
import torch
from scipy.ndimage import zoom
from skimage import measure

from models import ModelLoader
from utils import BoundryFinder

MODEL_FILE = "/model/regression_model.pkl"
AC_MODLE_FILE = "/model/ac_model.pkl"
VESSEL_MODLE_FILE = "/model/vessel_model.pkl"

AC_SEG_NET_NAME = "SKUNET2_DICE"
VESSEL_SEG_NET_NAME = "SKUNET2_DICE_BN"


class Segmentor(object):
    def __init__(self, input_npy_file, output_dir, first_model_file, second_model_file):
        self.ac_model_file = first_model_file
        self.vessel_model_file = second_model_file
        self.ori_input_file = input_npy_file
        self.output_dir = output_dir
        self.modified_npy = None
        self.seg_result_npy = None
        self.pt1 = None
        self.pt2 = None

    def cut_npy(self):
        if self.modified_npy is None:
            ori_npy = np.load(self.ori_input_file)
            pt1, pt2 = BoundryFinder.find_by_regression(MODEL_FILE, self.ori_input_file)
            self.pt1 = pt1
            self.pt2 = pt2

            cz = np.array([320, 320, 320])
            scale = np.true_divide(cz, (pt2 - pt1))

            ndata = zoom(ori_npy, scale, order=1)
            ndata = ndata.astype("int16")
            pt1 = pt1 * scale
            pt2 = pt2 * scale

            zstart = int(pt1[0])
            zend = int(pt2[0])
            if zstart < 0:
                zstart = 0
                zend = cz.shape[0]
            if zend >= ndata.shape[0]:
                zend = ndata.shape[0] - 1
                zstart = zend - cz[0]

            ndata_cut = ndata[
                zstart:zend,
                int(pt1[1]) : int(pt1[1]) + cz[1],
                int(pt1[2]) : int(pt1[2]) + cz[2],
            ]
            self.modified_npy = ndata_cut

        return self.modified_npy

    def _expandPt(self, pt1, pt2, expend, imshape):
        pt1 = pt1 - expend
        pt2 = pt2 + expend

        for j in range(3):
            if pt1[j] < 0:
                pt1[j] = 0
            if pt2[j] > imshape[j] - 1:
                pt2[j] = imshape[j] - 1

        return pt1, pt2

    def _cleanNoisePred(self, label):
        labelb = label.copy().astype("bool")

        label_ids = measure.label(labelb, neighbors=8, connectivity=2)
        info_label = measure.regionprops(label_ids)

        #    print('label numbers: ', len(info_label))

        count = 0
        for i in range(len(info_label)):
            if info_label[i].area <= 50:
                label_ids[label_ids == info_label[i].label] = 0

        label_ids[label_ids > 0] = 1

        return label_ids

    def _seg(self, net_name, model_file):
        self.cut_npy()
        print("running")
        net_t = ModelLoader.load(net_name)
        net_t = torch.nn.DataParallel(net_t)
        net_t = net_t.eval()
        state = torch.load(model_file)
        cropsize = np.array([64, 320, 320])
        stride_z = 64

        net_t.load_state_dict(state)
        net_t = torch.nn.DataParallel(net_t).cuda()

        sigmoid = torch.nn.Sigmoid()
        img = self.modified_npy

        img = (img + 300.0) / 800.0
        img[img > 1] = 1.0
        img[img < 0] = 0.0

        pred = img.copy()
        pred[pred != 0] = 0

        lpt1 = self.pt1
        lpt2 = self.pt2

        expend = np.array([10, 10, 10])
        lpt1, lpt2 = self._expandPt(lpt1, lpt2, expend, img.shape)

        num = int(img.shape[0] / stride_z)
        if num != img.shape[0] / stride_z:
            num = num + 1

        for i in range(num):

            lpt2_e = lpt2.copy()
            lpt2_e[1] = lpt1[1] + cropsize[1]
            lpt2_e[2] = lpt1[2] + cropsize[2]
            if lpt2_e[1] > img.shape[1]:
                lpt2_e[1] = img.shape[1]
                lpt1[1] = lpt2_e[1] - cropsize[1]
            if lpt2_e[2] > img.shape[2]:
                lpt2_e[2] = img.shape[2]
                lpt1[2] = lpt2_e[2] - cropsize[2]

            start_z = lpt1[0] + stride_z * i
            end_z = lpt1[0] + stride_z * (i + 1)
            if end_z > img.shape[0]:
                end_z = img.shape[0]
                start_z = end_z - cropsize[0]

            imgc = img[start_z:end_z, lpt1[1] : lpt2_e[1], lpt1[2] : lpt2_e[2]]

            print("imshape:", imgc.shape)

            imgc = imgc.astype("float32")
            imgcth = torch.autograd.Variable(
                torch.from_numpy(imgc[np.newaxis, np.newaxis]).cuda(), volatile=True
            )

            output = net_t(imgcth)
            Sigmoidout = sigmoid(output)
            prob5_np = Sigmoidout.data.cpu().numpy()

            prob_np = prob5_np[0][0]

            pred[start_z:end_z, lpt1[1] : lpt2_e[1], lpt1[2] : lpt2_e[2]] += prob_np

        print("pred", pred.shape)

        pred[pred > 0.15] = 1
        pred[pred <= 0.15] = 0
        pred = self._cleanNoisePred(pred)
        return pred

    def do_ac_seg(self):
        return self._seg(AC_SEG_NET_NAME, self.ac_model_file)

    def do_vessel_seg(self):
        return self._seg(VESSEL_SEG_NET_NAME, self.vessel_model_file)

    def _mergeImg(self, img_ac, img_noac):
        img_whole = img_ac + img_noac
        img_whole[img_whole > 1] = 1
        img_whole[img_whole < 1] = 0
        return img_whole

    def _zoomAndShift(self, img_wholepred, pt1, pt2, crop_size):
        im_pred_final = np.full([260, 512, 512], 0.0)
        zoomfactor = (pt2 - pt1) / crop_size
        img_wholepred_zoomori = zoom(img_wholepred, zoomfactor, mode="nearest", order=1)
        img_wholepred_zoomori[img_wholepred_zoomori > 0.5] = 1
        img_wholepred_zoomori[img_wholepred_zoomori <= 0.5] = 0
        wid = img_wholepred_zoomori.shape
        im_pred_final[
            pt1[0] : pt1[0] + wid[0], pt1[1] : pt1[1] + wid[1], pt1[2] : pt1[2] + wid[2]
        ] = img_wholepred_zoomori[:, :, :]

        return im_pred_final

    def merge_seg(self, img_ac, img_noac):
        crop_size = np.array([320, 320, 320])

        img_wholepred = self._mergeImg(img_ac, img_noac)
        im_pred_final = self._zoomAndShift(img_wholepred, self.pt1, self.pt2, crop_size)

        print(im_pred_final.shape)
        return im_pred_final

    def doseg(self):
        start1 = time.time()
        ac = self.do_ac_seg()
        end = time.time()
        print("seg ac completed in %.4fs" % (end - start1))

        start = time.time()
        vessel = self.do_vessel_seg()
        end = time.time()
        print("seg vessel completed in %.4fs" % (end - start))

        self.seg_result_npy = self.merge_seg(ac, vessel)
        end = time.time()
        print("all completed in %.4fs" % (end - start1))
        return self.seg_result_npy

    def generate_2d_result(self, out_path):
        import cv2

        data = self.seg_result_npy
        data[data > 0.2] = 255
        data[data <= 0.2] = 0

        data_png = np.full([data.shape[1], data.shape[2], 4], 0)
        data_png = data_png.astype("uint8")
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        num = data.shape[0]
        for i in range(num):
            data_i = data[num - i - 1]

            data_png[:, :, 0] = 0
            data_png[:, :, 1] = 0
            data_png[:, :, 2] = data_i
            data_png[:, :, 3] = data_i

            # use reverse order
            index = num - i - 1
            filename = os.path.join(out_path, "%.4d.png" % index)
            cv2.imwrite(filename, data_png)
        print("png files generated to %s" % out_path)


if __name__ == "__main__":
    input_npy = sys.argv[1]
    case_output = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # input_npy = '/tmp/case.npy'

    seg_gen = Segmentor(
        input_npy_file=input_npy,
        output_dir=case_output,
        first_model_file=AC_MODLE_FILE,
        second_model_file=VESSEL_MODLE_FILE,
    )
    seg_gen.doseg()
    seg_gen.generate_2d_result(os.path.join(case_output, "seg"))
