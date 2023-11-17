import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import net


def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default = 'input/content/1.jpg',
                    help='File path to the content image')
parser.add_argument('--style', type=str, default = 'input/style/1.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--steps', type=str, default = 1)
parser.add_argument('--vgg', type=str, default = 'model/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default = 'model/decoder_iter_160000.pth')
parser.add_argument('--transform', type=str, default = 'model/transformer_iter_160000.pth')
parser.add_argument('--featureFusion', type=str, default = 'model/featureFusion_iter_160000.pth')

# Additional options
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
transform = net.AdaptiveMultiAttn_Transformer_v2(in_planes = 512)
vgg = net.vgg
featureFusion = net.FeatureFusion

decoder.eval()
transform.eval()
vgg.eval()
featureFusion.eval()

decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
vgg.load_state_dict(torch.load(args.vgg))
featureFusion.load_state_dict(torch.load(args.featureFusion))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1


def encode_with_intermediate(input):
    results = [input]
    for i in range(5):
        func = getattr('enc_{:d}'.format(i + 1))
        results.append(func(results[-1]))
    return results[1:]

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform()

content = content_tf(Image.open(args.content))
style = style_tf(Image.open(args.style))

style = style.to(device).unsqueeze(0)
content = content.to(device).unsqueeze(0)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def adaptive_get_keys(feats, start_layer_idx, last_layer_idx):
    results = []
    _, _, h, w = feats[last_layer_idx].shape
    for i in range(start_layer_idx, last_layer_idx):
        results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
    results.append(mean_variance_norm(feats[last_layer_idx]))
    return torch.cat(results, dim=1)

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content, style, style_strength=1.0, eps=1e-5):
        b, c, h, w = content.size()

        content_std, content_mean = torch.std_mean(content.view(b, c, -1), dim=2, keepdim=True)
        style_std, style_mean = torch.std_mean(style.view(b, c, -1), dim=2, keepdim=True)

        normalized_content = (content.view(b, c, -1) - content_mean) / (content_std + eps)

        stylized_content = (normalized_content * style_std) + style_mean

        output = (1 - style_strength) * content + style_strength * stylized_content.view(b, c, h, w)
        return output

with torch.no_grad():

    for x in range(args.steps):

        print('iteration ' + str(x))

        style_feats = encode_with_intermediate(style)
        content_feats = encode_with_intermediate(content)

        content = decoder(featureFusion(transform(
            content_feats[3], style_feats[3], content_feats[4], style_feats[4],
            adaptive_get_keys(content_feats, 3, 3),
            adaptive_get_keys(style_feats, 0, 3),
            adaptive_get_keys(content_feats, 4, 4),
            adaptive_get_keys(style_feats, 0, 4))[0], AdaIN(content_feats[3], content_feats[3])))

        content.clamp(0, 255)

    content = content.cpu()
    
    output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                args.output, splitext(basename(args.content))[0],
                splitext(basename(args.style))[0], args.save_ext
            )
    save_image(content, output_name)
