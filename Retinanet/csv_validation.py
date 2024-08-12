import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval

#assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--images_path',help='Path to images directory',type=str)
    parser.add_argument('--class_list_path',help='Path to classlist csv',type=str)
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
    parser = parser.parse_args(args)

    #dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(parser.csv_annotations_path,parser.class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))
    # Create the model
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True) # To change according to the Resnet used eitheir to 50, 18, 150 etc..
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(),)
    retinanet.load_state_dict(torch.load(parser.model_path)["model_state_dict"])
    #retinanet=torch.load(parser.model_path)
    """
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in retinanet.state_dict():
        print(param_tensor, "\t", retinanet.state_dict()[param_tensor].size())

    """
   
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        """
        state_dict =retinanet.load_state_dict(checkpoint['model_state_dict'])
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k]=v
        retinanet.load_state_dict(new_state_dict)
        """
        #retinanet= torch.load(parser.model_path)
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    print(csv_eval.evaluate_new(dataset_val, retinanet,iou_threshold=float(parser.iou_threshold)))



if __name__ == '__main__':
    main()
