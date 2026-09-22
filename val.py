import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from tqdm import tqdm

# pip install pysodmetrics   （PyPI 发行名为 pysodmetrics，导入名为 py_sod_metrics）
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='EATNet',
                    help='results/ 下的方法目录名，须与 etest.py 的输出目录一致')
parser.add_argument('--datasets', type=str, nargs='+',
                    default=['CAMO', 'CHAMELEON', 'COD10K', 'NC4K'],
                    help='要评估的测试集')
opt = parser.parse_args()

method = opt.method
for _data_name in opt.datasets:
    mask_root = './data/TestDataset/{}/GT'.format(_data_name)
    pred_root = './results/{}/{}/'.format(method, _data_name)
    mask_name_list = sorted(os.listdir(mask_root))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    # pysodmetrics 返回的是 np.float64，直接 str() 会写成 "np.float64(0.86…)"。
    # 统一转成 Python float，保持 evalresults.txt 的格式干净。
    results = {
        "Smeasure": float(sm),
        "wFmeasure": float(wfm),
        "MAE": float(mae),
        "adpEm": float(em["adp"]),
        "meanEm": float(em["curve"].mean()),
        "maxEm": float(em["curve"].max()),
        "adpFm": float(fm["adp"]),
        "meanFm": float(fm["curve"].mean()),
        "maxFm": float(fm["curve"].max()),
    }

    print(results)
    file = open("evalresults.txt", "a")
    file.write(method + ' ' + _data_name + ' ' + str(results) + '\n')
