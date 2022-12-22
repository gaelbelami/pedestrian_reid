import argparse

# Construct the argument parse and pars the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--use-gpu", type=bool, default=0, help="boolean indicating if CUDA GPU should be used")
ap.add_argument('--min_score', type=float, default=0.3, help="displays the lowest tracking score")
ap.add_argument('--model_feature', type=str, default='model_data/market1501.pb', help='target tracking model')
# ap.add_argument('--input_size', type=int, default=1024, help='input pic size')
ap.add_argument("-d", "--dataset", default='dataset/hog',  help="Path to the directory of images")
args = vars(ap.parse_args())