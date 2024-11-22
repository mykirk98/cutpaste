# head dims:512,512,512,512,512,512,512,512,128
# code is basicly:https://github.com/google-research/deep_representation_one_class
from pathlib import Path
from tqdm import tqdm
import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from dataset import MVTecAT, Repeat
from cutpaste import CutPasteNormal,CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn
from model import ProjectionNet
from eval import eval_model
from utils import str2bool

from colors import *
import json
from parser_ import parse_args

def gpu_check():
    if torch.cuda.is_available() == True:
        print(f"CUDA available : {blue(torch.cuda.is_available())}")
        print(f"PyTorch version: {highlight(torch.__version__)}")
        print(f"CUDA device count: {highlight(torch.cuda.device_count())}")
        print(f"CUDA current device index: {highlight(torch.cuda.current_device())}")
        print(f"CUDA device name: {highlight(torch.cuda.get_device_name(0))}")
    else:
        print(f"CUDA available : {red({torch.cuda.is_available()})}")

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_config(config_file):
    with open(file=config_file, mode="r") as file:
        config = json.load(file)
    return config

def run_training(data_type, hyperParams, args, device, cutpaste_type, size=256):
    
    epochs, lr, batch_size, workers, optimizer, temperature, weight_decay, momentum = hyperParams.values()
    model_dir = Path(args.model_dir)
    test_epochs = args.test_epochs
    freeze_resnet = args.freeze_resnet
    head_layer = args.head_layer
    pretrained = str2bool(args.pretrained)
    
    
    
    
    # model_dir
    # torch.multiprocessing.freeze_support()
    # TODO: use script params for hyperparameter
    # Temperature Hyperparameter currently not used
    
    model_name = f"model-{data_type}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())

    #augmentation:
    min_scale = 1

    # create Training Dataset and Dataloader
    after_cutpaste_transform = transforms.Compose(transforms=[
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                        std=(0.229, 0.224, 0.225))
                                                            ])

    train_transform = transforms.Compose(transforms=[
                                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                            transforms.Resize(size=(size, size)),
                                            cutpaste_type(transform=after_cutpaste_transform)
                                                    ])
    # train_transform = transforms.Compose(transforms=[])
    # #train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
    # train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    # # train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
    # train_transform.transforms.append(transforms.Resize((size,size)))
    # train_transform.transforms.append(cutpaste_type(transform=after_cutpaste_transform))
    # # train_transform.transforms.append(transforms.ToTensor())

    train_data = MVTecAT("/home/msis/Work/dataset/mvtec", data_type, transform = train_transform, size=int(size * (1/min_scale)))
    dataloader = DataLoader(Repeat(train_data, 3000), batch_size=batch_size, drop_last=True, shuffle=True, num_workers=workers,
                            collate_fn=cut_paste_collate_fn, persistent_workers=True, pin_memory=True, prefetch_factor=5)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(Path("logdirs") / model_name)

    # create Model:
    head_layers = [512]*head_layer+[128]
    num_classes = 2 if cutpaste_type is not CutPaste3Way else 3
    # model = ProjectionNet(pretrained=pretrained, head_layers=head_layers, num_classes=num_classes)
    model = ProjectionNet(weights=pretrained, head_layers=head_layers, num_classes=num_classes)
    model.to(device)

    if freeze_resnet > 0 and pretrained:
        model.freeze_resnet()

    loss_fn = torch.nn.CrossEntropyLoss()
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,  weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)
        #scheduler = None
    elif optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = None
    else:
        print(f"ERROR unkown optimizer: {optim}")

    step = 0
    num_batches = len(dataloader)
    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
                yield out
    dataloader_inf =  get_data_inf()
    # From paper: "Note that, unlike conventional definition for an epoch,
    #              we define 256 parameter update steps as one epoch.
    for step in tqdm(range(epochs)):
        epoch = int(step / 1)
        if epoch == freeze_resnet:
            model.unfreeze()
        
        batch_embeds = []
        batch_idx, data = next(dataloader_inf)
        xs = [x.to(device) for x in data]

        # zero the parameter gradients
        optimizer.zero_grad()

        xc = torch.cat(xs, axis=0)
        embeds, logits = model(xc)
        
#         embeds = F.normalize(embeds, p=2, dim=1)
#         embeds1, embeds2 = torch.split(embeds,x1.size(0),dim=0)
#         ip = torch.matmul(embeds1, embeds2.T)
#         ip = ip / temperature

#         y = torch.arange(0,x1.size(0), device=device)
#         loss = loss_fn(ip, torch.arange(0,x1.size(0), device=device))

        # calculate label
        y = torch.arange(len(xs), device=device)
        y = y.repeat_interleave(xs[0].size(0))
        loss = loss_fn(logits, y)


        # regulize weights:
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch)
        
        writer.add_scalar('loss', loss.item(), step)
        
#         predicted = torch.argmax(ip,axis=0)
        predicted = torch.argmax(logits,axis=1)
#         print(logits)
#         print(predicted)
#         print(y)
        accuracy = torch.true_divide(torch.sum(predicted==y), predicted.size(0))
        writer.add_scalar('acc', accuracy, step)
        if scheduler is not None:
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)
        
        # save embed for validation:
        if test_epochs > 0 and epoch % test_epochs == 0:
            batch_embeds.append(embeds.cpu().detach())

        writer.add_scalar('epoch', epoch, step)

        # run tests
        if test_epochs > 0 and epoch % test_epochs == 0:
            # run auc calculation
            #TODO: create dataset only once.
            #TODO: train predictor here or in the model class itself. Should not be in the eval part
            #TODO: we might not want to use the training datat because of droupout etc. but it should give a indecation of the model performance???
            # batch_embeds = torch.cat(batch_embeds)
            # print(batch_embeds.shape)
            model.eval()
            roc_auc= eval_model(model_name, data_type, device=device, save_plots=False,
                                size=size, show_training_data=False, model=model)
            model.train()
            writer.add_scalar('eval_auc', roc_auc, step)


    torch.save(model.state_dict(), model_dir / f"{model_name}.tch")

if __name__ == '__main__':
    device = gpu_check()
    
    args = parse_args()
    hyperParams = load_config(config_file="hyperparameter.json")
    config = load_config(config_file="config.json")
    class_ = config['classes']
    
    if args.type == "all":
        types = class_
    else:
        types = args.type.split(",")
    
    variant = config['cutpaste']['variant']['normal']
    if variant == 'CutPasteNormal':
        variant = CutPasteNormal
    elif variant == 'CutPasteScar':
        variant = CutPasteNormal
    elif variant == 'CutPaste3Way':
        variant = CutPaste3Way
    elif variant == 'CutPasteUnion':
        variant = CutPasteUnion
    
    # create modle dir
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # save config.
    with open(Path(args.model_dir) / "run_config.txt", "w") as f:
        f.write(str(args))

    for data_type in types:
        print(f"training {yellow(data_type)}")
        # run_training(data_type=data_type,
        #                 model_dir=Path(args.model_dir),
        #                 pretrained=args.pretrained,
        #                 test_epochs=args.test_epochs,
        #                 freeze_resnet=args.freeze_resnet,
        #                 head_layer=args.head_layer,
        #                 device=device,
        #                 cutpate_type=variant)
        run_training(data_type=data_type, device=device, hyperParams=hyperParams, args=args, cutpaste_type=variant)