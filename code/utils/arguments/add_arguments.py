import argparse

def add_arguments():
    ap = argparse.ArgumentParser(prog='Robust Collaborative Multi-label Learning', 
            description='Robust Collaborative Multi-label Learning')
    ap.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size. Default is 64.')
    ap.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs. Default is 10.')
    ap.add_argument('-l', '--loss_fn', default='BEN', help='Loss function to be used. Default is focal loss.')
    ap.add_argument('-a', '--architecture', default='resnet', help='Possible architectures: \
        resnet, denseNet, SCNN, paper_model, keras_model, modified_SCNN, batched_SCNN. Default is resnet.')
    ap.add_argument('-d', '--dataset_path', help='Give the path to the folder where tf.record files are located.')
    ap.add_argument('-ch', '--channels', type=int, default=3, help='Number of channels in the image.')
    ap.add_argument('-lb', '--label_type', default='BEN-12', help='Decide what version of BigEarthNet \
        you want to load, or load UC Merced Land Use data set. Possible values are \
            BEN-12, BEN-19, BEN-43, ucmerced.')
    ap.add_argument('-si', '--sigma', type=float, help='The value of the sigma for the gaussian kernel.')
    ap.add_argument('-sw', '--swap', type=int, help='Swap information between models if 1; if 0 do not swap')
    ap.add_argument('-lto', '--lambda_two', type=float, help='Strength of discrepancy component. \
        Give a value between 0. and 1.')
    ap.add_argument('-ltr', '--lambda_three', type=float, help='Strength of consistency component. \
        Give a value between 0. and 1.')
    ap.add_argument('-ma', '--miss_alpha', type=float, help='Rate of miss noise to be included into the error loss')
    ap.add_argument('-eb', '--extra_beta', type=float, help='Rate of extra noise to be included into the error loss')
    ap.add_argument('-sar', '--sample_rate', type=float, help='Percentage of samples in a mini-batch \
        to be noisified. Give a value between 0. and 1.')
    ap.add_argument('-car', '--class_rate', type=float, help='Percentage of labels in a sample to be noisified. \
        Mix Label Noise does not use this. Give a value between 0. and 1.')
    ap.add_argument('-dm', '--divergence_metric', help='Possible metrics to diverge and converge the models: \
        mmd, shannon, wasserstein, nothing')
    ap.add_argument('-test', '--test', type=int, default=0, help='Enter 1 to test the model using \
        only a small portion of the datasets. Default is 0.')
    ap.add_argument('-gpu', '--gpu', type=int, default=1, help='Enter 1 to use gpu; 0 to use cpu.')
    ap.add_argument('-logname', '--logname', help='Enter a string for the name of the log file.')
    ap.add_argument('-swap_start', '--swap_start', type=int, help='Start swapping with the given rate over 100.')
    ap.add_argument('-swap_end', '--swap_end', type=int, help='End swapping with the given rate over 100.')
    ap.add_argument('-prediction_threshold', '--prediction_threshold', type=float, help='Threshold for prediction.')
    ap.add_argument('-base', '--base', type=int, default=0,  help='If you want to run a single model to run BCE, FL, SAT and ELR, set this to 1. \
            If you are running RCML, set this to 0.')
    ap.add_argument('-seed', '--seed', type=int, default=1,  help='Give an integer seed.')
    ap.add_argument('-random_instead_lasso', '--random_instead_lasso', type=int, default=0,  help='Do not use group lasso module, instead exclude \
            samples randomly. This is for testing the ability of the group lasso module.')
    args = ap.parse_args()

    return args
