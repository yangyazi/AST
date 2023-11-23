import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import net


# def test_transform():
#     transform_list = []
#     transform_list.append(transforms.ToTensor())
#     transform = transforms.Compose(transform_list)
#     return transform


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')

#######
parser.add_argument('--content_dir', type=str,default = r'E:\00propropropro\pytorch-AdaIN-master\pytorch-AdaIN-master\input\content',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str,default = r'E:\00propropropro\pytorch-AdaIN-master\pytorch-AdaIN-master\input\style',
                    help='Directory path to a batch of style images')
#######


parser.add_argument('--steps', type=str, default = 1)
parser.add_argument('--vgg', type=str, default = 'model/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default = 'model/decoder_iter_160000.pth')
parser.add_argument('--transform', type=str, default = 'model/transformer_iter_160000.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=0,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')

parser.add_argument('--save_ext', default = '.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default = 'output',
                    help='Directory to save the output image(s)')

# Advanced options

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# device = torch.device('cuda')
# print(device)
# print(os.environ['CUDA_VISIBLE_DEVICES'])

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net.decoder
transform = net.Transform(in_planes = 512)
vgg = net.vgg

decoder.eval()
transform.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
vgg.load_state_dict(torch.load(args.vgg))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)

# content_tf = test_transform()
# style_tf = test_transform()

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

output_dir = Path(args.output)
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)

if args.style:
    style_paths = [Path(args.style)]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]


for content_path in content_paths:

    for style_path in style_paths:
        content = content_tf(Image.open(str(content_path)))
        style = style_tf(Image.open(str(style_path)))

        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            for x in range(args.steps):
                print('iteration ' + str(x))

                Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
                Content5_1 = enc_5(Content4_1)

                Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
                Style5_1 = enc_5(Style4_1)

                content = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))

                content.clamp(0, 255)

            content = content.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(content, str(output_name))

####################################