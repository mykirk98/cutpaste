import argparse
from colors import yellow, magenta


def parse_args():
    """
    parse command line arguments
    """
    
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', dest='type', default="all", help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')
    parser.add_argument('--model_dir', dest='model_dir', default="models", help='output folder of the models , (default: models)')
    parser.add_argument('--test_epochs', dest='test_epochs', default=10, type=int, help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')
    parser.add_argument('--freeze_resnet', dest='freeze_resnet', default=20, type=int, help='number of epochs to freeze resnet (default: 20)')
    parser.add_argument('--head_layer', dest='head_layer', default=1, type=int, help='number of layers in the projection head (default: 1)')
    parser.add_argument('--pretrained', dest='pretrained', default=True, help='use pretrained values to initalize ResNet18 , (default: True)')
    # parser.add_argument('--epochs', dest='epochs', default=256, type=int,
    #                     help='number of epochs to train the model , (default: 256)')
    # parser.add_argument('--lr', dest='lr', default=0.03, type=float,
    #                     help='learning rate (default: 0.03)')
    # parser.add_argument('--optim', dest='optim', default="sgd",
    #                     help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')
    # parser.add_argument('--batch_size', dest='batch_size', default=64, type=int,
                        # help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')
    # parser.add_argument('--workers', dest='workers', default=8, type=int, help="number of workers to use for data loading (default:8)")

    args = parser.parse_args()
    print(f"type : {yellow(args.type)}")
    print(f"model_dir : {yellow(args.model_dir)}")
    print(f"test_epochs : {magenta(args.test_epochs)}")
    print(f"freeze_resnet : {yellow(args.freeze_resnet)}")
    print(f"head_layer : {yellow(args.head_layer)}")
    # print(f"epochs : {magenta(args.epochs)}")
    # print(f"lr : {magenta(args.lr)}")
    # print(f"optim : {yellow(args.optim)}")
    # print(f"batch_size : {magenta(args.batch_size)}")
    # print(f"workers : {magenta(args.workers)}")
    
    return args