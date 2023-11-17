# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:35:21 2020

@author: ZJU
"""

import torch
import torch.nn as nn


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

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

# projection_style = nn.Sequential(
#     nn.Linear(in_features=256, out_features=128),
#     nn.ReLU(),
#     nn.Linear(in_features=128, out_features=128)
# )
#
# projection_content = nn.Sequential(
#     nn.Linear(in_features=512, out_features=256),
#     nn.ReLU(),
#     nn.Linear(in_features=256, out_features=128)
# )


class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


# class SANet(nn.Module):
#     def __init__(self, in_planes):
#         super(SANet, self).__init__()
#         self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
#         self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
#         self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
#         self.sm = nn.Softmax(dim = -1)
#         self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
#     def forward(self, content, style):
#         F = self.f(mean_variance_norm(content))
#         G = self.g(mean_variance_norm(style))
#         H = self.h(style)
#         b, c, h, w = F.size()
#         F = F.view(b, -1, w * h).permute(0, 2, 1)
#         b, c, h, w = G.size()
#         G = G.view(b, -1, w * h)
#         S = torch.bmm(F, G)
#         S = self.sm(S)
#         b, c, h, w = H.size()
#         H = H.view(b, -1, w * h)
#         O = torch.bmm(H, S.permute(0, 2, 1))
#         b, c, h, w = content.size()
#         O = O.view(b, c, h, w)
#         O = self.out_conv(O)
#         O += content
#         return O

# class Transform(nn.Module):
#     def __init__(self, in_planes):
#         super(Transform, self).__init__()
#         self.sanet4_1 = SANet(in_planes = in_planes)
#         self.sanet5_1 = SANet(in_planes = in_planes)
#         #self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
#         self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
#     def forward(self, content4_1, style4_1, content5_1, style5_1):
#         self.upsample5_1 = nn.Upsample(size=(content4_1.size()[2], content4_1.size()[3]), mode='nearest')
#         return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))


class AdaptiveMultiAdaAttN_v2(nn.Module):
    def __init__(self, in_planes, out_planes, max_sample=256 * 256, query_planes=None, key_planes=None):
        super(AdaptiveMultiAdaAttN_v2, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(query_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, out_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, out_planes, (1, 1))
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)

        b, style_c, style_h, style_w = H.size()
        H = torch.nn.functional.interpolate(H, (h_g, w_g), mode='bicubic')
        if h_g * w_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(h_g * w_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, h_g * w_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, h_g * w_g).transpose(1, 2).contiguous()

        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        _, _, ch, cw = content.size()
        # mean = torch.nn.functional.interpolate(mean.view(b, style_h, style_w, style_c).permute(0, 3, 1, 2).contiguous(), (ch, cw))
        # std = torch.nn.functional.interpolate(std.view(b, style_h, style_w, style_c).permute(0, 3, 1, 2).contiguous(), (ch, cw))
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean, S

class AdaptiveMultiAttn_Transformer_v2(nn.Module):
    def __init__(self, in_planes, out_planes, query_planes=None, key_planes=None, shallow_layer=False):
        super(AdaptiveMultiAttn_Transformer_v2, self).__init__()
        self.attn_adain_4_1 = AdaptiveMultiAdaAttN_v2(in_planes=in_planes, out_planes=out_planes,
                                                      query_planes=query_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaptiveMultiAdaAttN_v2(in_planes=in_planes, out_planes=out_planes,
                                                      query_planes=query_planes, key_planes=key_planes + 512)
        # self.attn_adain_5_1 = AdaptiveMultiAdaAttN_v2(in_planes=in_planes, out_planes=out_planes, query_planes=query_planes+512, key_planes=key_planes+512)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(out_planes, out_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key, content5_1_key,
                style5_1_key, seed=None):
        feature_4_1, attn_4_1 = self.attn_adain_4_1(content4_1, style4_1, content4_1_key, style4_1_key, seed=seed)
        feature_5_1, attn_5_1 = self.attn_adain_5_1(content5_1, style5_1, content5_1_key, style5_1_key, seed=seed)
        # stylized_results = self.merge_conv(self.merge_conv_pad(feature_4_1 +  self.upsample5_1(feature_5_1)))

        stylized_results = self.merge_conv(self.merge_conv_pad(
            feature_4_1 + nn.functional.interpolate(feature_5_1, size=(feature_4_1.size(2), feature_4_1.size(3)))))
        return stylized_results, feature_4_1, feature_5_1, attn_4_1, attn_5_1


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


class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super(FeatureFusion, self).__init__()
        self.proj1 = nn.Conv2d(2 * channels, channels, (1, 1))
        self.proj2 = nn.Conv2d(channels, channels, (1, 1))
        self.proj3 = nn.Conv2d(channels, channels, (1, 1))

        # Fuse features with different kernel sizes
        self.fuse1x = nn.Conv2d(channels, 1, (1, 1), stride=1)
        self.fuse3x = nn.Conv2d(channels, 1, (3, 3), stride=1)
        self.fuse5x = nn.Conv2d(channels, 1, (5, 5), stride=1)

        # Padding layers
        self.pad3x = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pad5x = nn.ReflectionPad2d((2, 2, 2, 2))

        self.sigmoid = nn.Sigmoid()

    def forward(self, global_feature, local_feature):
        Fcat = self.proj1(torch.cat((global_feature, local_feature), dim=1))
        global_feature = self.proj2(global_feature)
        local_feature = self.proj3(local_feature)

        # Get fusion weights from different kernel sizes
        fusion1 = self.sigmoid(self.fuse1x(Fcat))
        fusion3 = self.sigmoid(self.fuse3x(self.pad3x(Fcat)))
        fusion5 = self.sigmoid(self.fuse5x(self.pad5x(Fcat)))

        # Average fusion weights
        fusion = (fusion1 + fusion3 + fusion5) / 3

        # Fuse global and local features with the fusion weights
        return fusion * global_feature + (1 - fusion) * local_feature

class Net(nn.Module):
    def __init__(self, encoder, decoder, start_iter):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        #projection
        # self.proj_style = projection_style
        # self.proj_content = projection_content

        #transform
        # self.transform = Transform(in_planes = 512)
        self.decoder = decoder
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        query_channels = 512  # +256+128+64
        key_channels = 512 + 256 + 128 + 64
        self.transformer = AdaptiveMultiAttn_Transformer_v2(in_planes=512, out_planes=512, query_planes=query_channels,
                                                            key_planes=key_channels)
        self.adain = AdaIN()
        self.featureFusion = FeatureFusion(512)

        if(start_iter > 0):
            self.transform.load_state_dict(torch.load('transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(torch.load('decoder_iter_' + str(start_iter) + '.pth'))
            self.featureFusion.load_state_dict('featureFusion_iter' + str(start_iter) + '.pth')
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # def adaptive_get_keys(self, feat_skips, start_layer_idx, last_layer_idx, target_feat):
    #     B, C, th, tw = target_feat.shape
    #     results = []
    #     target_conv_layer = 'conv' + str(last_layer_idx) + '_1'
    #     _, _, h, w = feat_skips[target_conv_layer].shape
    #     for i in range(start_layer_idx, last_layer_idx + 1):
    #         target_conv_layer = 'conv' + str(i) + '_1'
    #         if i == last_layer_idx:
    #             results.append(mean_variance_norm(feat_skips[target_conv_layer]))
    #         else:
    #             results.append(mean_variance_norm(nn.functional.interpolate(feat_skips[target_conv_layer], (h, w))))
    #
    #     return nn.functional.interpolate(torch.cat(results, dim=1), (th, tw))


    def adaptive_get_keys(self, feats, start_layer_idx, last_layer_idx):

        results = []
        _, _, h, w = feats[last_layer_idx].shape
        for i in range(start_layer_idx, last_layer_idx):
            results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
        results.append(mean_variance_norm(feats[last_layer_idx]))
        return torch.cat(results, dim=1)


    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
          return self.mse_loss(input, target)
        else:
          return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        #loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        loss = self.cross_entropy_loss(out, torch.tensor([index], dtype=torch.long, device=feat_q.device))
        return loss

    def style_feature_contrastive(self, input):
        # out = self.enc_style(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_style(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out

    def content_feature_contrastive(self, input):
        #out = self.enc_content(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_content(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out
    
    def forward(self, content, style, batch_size):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)

        local_transformed_feature, attn_style_4_1, attn_style_5_1, attn_map_4_1, attn_map_5_1 = self.transformer(
            content_feats[3], style_feats[3], content_feats[4], style_feats[4],
            self.adaptive_get_keys(content_feats, 3, 3),
            self.adaptive_get_keys(style_feats, 0, 3),
            self.adaptive_get_keys(content_feats, 4, 4),
            self.adaptive_get_keys(style_feats, 0, 4))
        global_transformed_feat = self.adain(content_feats[3], style_feats[3])
        stylized = self.featureFusion(local_transformed_feature, global_transformed_feat)


        # stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        g_t = self.decoder(stylized)
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm = True) + self.calc_content_loss(g_t_feats[4], content_feats[4], norm = True)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        """IDENTITY LOSSES"""
        # Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        # Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))

        Icc = self.decoder(self.featureFusion(self.transformer(
            content_feats[3], content_feats[3], content_feats[4], content_feats[4],
            self.adaptive_get_keys(content_feats, 3, 3),
            self.adaptive_get_keys(content_feats, 0, 3),
            self.adaptive_get_keys(content_feats, 4, 4),
            self.adaptive_get_keys(content_feats, 0, 4))[0], self.adain(content_feats[3], content_feats[3])))
        Iss = self.decoder(self.featureFusion(self.transformer(
            style_feats[3], style_feats[3], style_feats[4], style_feats[4],
            self.adaptive_get_keys(style_feats, 3, 3),
            self.adaptive_get_keys(style_feats, 0, 3),
            self.adaptive_get_keys(style_feats, 4, 4),
            self.adaptive_get_keys(style_feats, 0, 4))[0], self.adain(style_feats[3], style_feats[3])))


        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i], style_feats[i])

        # # Contrastive learning.
        # half = int(batch_size / 2)
        # style_up = self.style_feature_contrastive(g_t_feats[2][0:half])
        # style_down = self.style_feature_contrastive(g_t_feats[2][half:])
        # content_up = self.content_feature_contrastive(g_t_feats[3][0:half])
        # content_down = self.content_feature_contrastive(g_t_feats[3][half:])
        #
        # style_contrastive_loss = 0
        # for i in range(half):
        #     reference_style = style_up[i:i+1]
        #
        #     if i ==0:
        #         style_comparisons = torch.cat([style_down[0:half-1], style_up[1:]], 0)
        #     elif i == 1:
        #         style_comparisons = torch.cat([style_down[1:], style_up[0:1], style_up[2:]], 0)
        #     elif i == (half-1):
        #         style_comparisons = torch.cat([style_down[half-1:], style_down[0:half-2], style_up[0:half-1]], 0)
        #     else:
        #         style_comparisons = torch.cat([style_down[i:], style_down[0:i-1], style_up[0:i], style_up[i+1:]], 0)
        #
        #     style_contrastive_loss += self.compute_contrastive_loss(reference_style, style_comparisons, 0.2, 0)
        #
        # for i in range(half):
        #     reference_style = style_down[i:i+1]
        #
        #     if i ==0:
        #         style_comparisons = torch.cat([style_up[0:1], style_up[2:], style_down[1:]], 0)
        #     elif i == (half-2):
        #         style_comparisons = torch.cat([style_up[half-2:half-1], style_up[0:half-2], style_down[0:half-2], style_down[half-1:]], 0)
        #     elif i == (half-1):
        #         style_comparisons = torch.cat([style_up[half-1:], style_up[1:half-1], style_down[0:half-1]], 0)
        #     else:
        #         style_comparisons = torch.cat([style_up[i:i+1], style_up[0:i], style_up[i+2:], style_down[0:i], style_down[i+1:]], 0)
        #
        #     style_contrastive_loss += self.compute_contrastive_loss(reference_style, style_comparisons, 0.2, 0)
        #
        #
        # content_contrastive_loss = 0
        # for i in range(half):
        #     reference_content = content_up[i:i+1]
        #
        #     if i == 0:
        #         content_comparisons = torch.cat([content_down[half-1:], content_down[1:half-1], content_up[1:]], 0)
        #     elif i == 1:
        #         content_comparisons = torch.cat([content_down[0:1], content_down[2:], content_up[0:1], content_up[2:]], 0)
        #     elif i == (half-1):
        #         content_comparisons = torch.cat([content_down[half-2:half-1], content_down[0:half-2], content_up[0:half-1]], 0)
        #     else:
        #         content_comparisons = torch.cat([content_down[i-1:i], content_down[0:i-1], content_down[i+1:], content_up[0:i], content_up[i+1:]], 0)
        #
        #     content_contrastive_loss += self.compute_contrastive_loss(reference_content, content_comparisons, 0.2, 0)
        #
        # for i in range(half):
        #     reference_content = content_down[i:i+1]
        #
        #     if i == 0:
        #         content_comparisons = torch.cat([content_up[1:], content_down[1:]], 0)
        #     elif i == (half-2):
        #         content_comparisons = torch.cat([content_up[half-1:], content_up[0:half-2], content_down[0:half-2], content_down[half-1:]], 0)
        #     elif i == (half-1):
        #         content_comparisons = torch.cat([content_up[0:half-1], content_down[0:half-1]], 0)
        #     else:
        #         content_comparisons = torch.cat([content_up[i+1:i+2], content_up[0:i], content_up[i+2:], content_down[0:i], content_down[i+1:]], 0)
        #
        #     content_contrastive_loss += self.compute_contrastive_loss(reference_content, content_comparisons, 0.2, 0)

        # return g_t, loss_c, loss_s, l_identity1, l_identity2, content_contrastive_loss, style_contrastive_loss
        return g_t, loss_c, loss_s, l_identity1, l_identity2