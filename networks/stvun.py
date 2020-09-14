import torch
import torch.nn.functional as F
from .submodules import *
from .pwcnet import *

class Encoder(torch.nn.Module):
    def __init__(self, nif=64):
        super(Encoder,self).__init__()

        self.conv = conv(3,   nif, kernel_size=3, stride=1)
        
        self.block0 = DenseBlock(nif, nif)
        self.block1 = DenseBlock(nif, nif)
        self.block2 = DenseBlock(nif, nif)
        self.block3 = DenseBlock(nif, nif)

    def forward(self, x):
        x = self.conv(x)
        e0 = self.block0(x)
        e1 = self.block1(e0)
        e2 = self.block2(e1)
        e3 = self.block3(e2)

        return [e3,e2,e1,e0]


class Decoder(torch.nn.Module):
    def __init__(self, nif=64, sf = 4):
        super(Decoder, self).__init__()
        
        self.sf = sf

        od = nif*2
        self.block0_0 = DenseBlock(od, od)
        self.block0_1 = DenseBlock(od*2, od*2)
        self.block0_2 = DenseBlock(od*4, nif)
        
        self.cconv1 = torch.nn.Conv2d(nif*2, nif, kernel_size=1, padding=0)
        self.block1_0 = DenseBlock(od, od)
        self.block1_1 = DenseBlock(od*2, od*2)
        self.block1_2 = DenseBlock(od*4, nif)
        
        self.cconv2 = torch.nn.Conv2d(nif*2, nif, kernel_size=1, padding=0)
        self.block2_0 = DenseBlock(od, od)
        self.block2_1 = DenseBlock(od*2, od*2)
        self.block2_2 = DenseBlock(od*4, nif)
        
        self.cconv3 = torch.nn.Conv2d(nif*2, nif, kernel_size=1, padding=0)
        self.block3_0 = DenseBlock(od, od)
        self.block3_1 = DenseBlock(od*2, od*2)
        self.block3_2 = DenseBlock(od*4, nif)
        
        self.rconv0 = conv(nif, 3*self.sf**2, kernel_size=3, padding=1) 
        self.rconv1 = torch.nn.Conv2d(3*self.sf**2,3*self.sf**2, kernel_size=3, padding=1)
    
    
    def forward(self, tsx, rsx):

        x = torch.cat((tsx[0], rsx[0]), 1)
        s0 = self.block0_0(x)
        s0 = torch.cat((x, s0), 1)
        s1 = self.block0_1(s0)
        s1 = torch.cat((s0, s1), 1)
        sx0 = self.block0_2(s1)
        
        trsx = self.cconv1(torch.cat((sx0, rsx[1]), 1) )
        x = torch.cat((tsx[1], trsx), 1)
        s0 = self.block1_0(x)
        s0 = torch.cat((x, s0), 1)
        s1 = self.block1_1(s0)
        s1 = torch.cat((s0, s1), 1)
        sx1 = self.block1_2(s1)
        
        trsx = self.cconv2(torch.cat((sx1, rsx[2]), 1) )
        x = torch.cat((tsx[2], trsx), 1)
        s0 = self.block2_0(x)
        s0 = torch.cat((x, s0), 1)
        s1 = self.block2_1(s0)
        s1 = torch.cat((s0, s1), 1)
        sx2 = self.block2_2(s1)
        
        trsx = self.cconv3(torch.cat((sx2, rsx[3]), 1) )
        x = torch.cat((tsx[3], trsx), 1)
        s0 = self.block3_0(x)
        s0 = torch.cat((x, s0), 1)
        s1 = self.block3_1(s0)
        s1 = torch.cat((s0, s1), 1)
        sx3 = self.block3_2(s1)
    
        res = self.rconv0(sx3)
        res = self.rconv1(res)
    

        return [sx0, sx1, sx2, sx3], res



class FeatureFusion(torch.nn.Module):
    def __init__(self, fn=3, nif=64):
        super(FeatureFusion, self).__init__()
        '''
        This module is from https://github.com/xinntao/EDVR
        Only the minor parts of spatial attention are modified

        '''

        
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)
    
        
        self.tAtt0 = torch.nn.Conv2d(nif, nif, kernel_size=3, padding=1)
        self.tAtt1 = torch.nn.Conv2d(nif, nif, kernel_size=3, padding=1)

        # spatial attention (after fusion conv)
        self.sAtt1 = conv(fn * nif, nif, kernel_size=1, padding=0)
        self.sAtt2 = conv(nif * 2, nif, kernel_size=1, padding=0)
        self.sAtt3 = conv(nif, nif, kernel_size=3, padding=1)
        self.sAtt4 = conv(nif, nif, kernel_size=1, padding=0)
        self.sAtt5 = torch.nn.Conv2d(nif, nif, kernel_size=3, padding=1)
        self.sAttL1 = conv(nif, nif, kernel_size=1, padding=0)
        self.sAttL2 = conv(nif * 2, nif, kernel_size=3, padding=1)
        self.sAttL3 = conv(nif, nif, kernel_size=3, padding=1)
        self.sAttA1 = conv(nif, nif, kernel_size=1, padding=0)
        self.sAttA2 = torch.nn.Conv2d(nif, nif, kernel_size=1,padding=0)

        

    def forward(self, ref_fea, target_fea):

        # shape of ref_fea : [B,N,C,H,W]
        # shape of target_fea : [B,C,H,W]

        B, N, C, H, W = ref_fea.size()  # N video frames

        # Temporal attention
        emb_ref = self.tAtt1(target_fea.clone())
        emb = self.tAtt0(ref_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        ref_fea = ref_fea.view(B, -1, H, W) * (cor_prob)

        
        # Spatial attention
        att = self.sAtt1(ref_fea)
        init_att = att
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.sAtt2(torch.cat([att_max, att_avg], dim=1))
        
        # Pyramid levels
        att_L = self.sAttL1(att)
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.sAttL2(torch.cat([att_max, att_avg], dim=1))
        att_L = self.sAttL3(att_L)
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.sAtt3(att)
        att = att + att_L
        att = self.sAtt4(att)
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = init_att + att
        att = self.sAtt5(att)
        att_add = self.sAttA2(self.sAttA1(att))
        att = torch.sigmoid(att)

        fea = target_fea * att * 2 + att_add

        return fea

class EFST(torch.nn.Module):
    def __init__(self, nif=64):
        super(EFST, self).__init__()
        
        self.ff = FeatureFusion(7, nif)
        self.cconv = torch.nn.Conv2d(nif*7, nif, kernel_size=1, padding=0)

    def forward(self , enc_fs):

        efst_outs = []
        for f in range(len(enc_fs[0])):

            enc_fs_stack = torch.stack((enc_fs[0][f],enc_fs[1][f],enc_fs[2][f],\
                enc_fs[3][f],enc_fs[4][f],enc_fs[5][f],enc_fs[6][f]), 1)
            
            enc_fs_concat = torch.cat((enc_fs[0][f],enc_fs[1][f],enc_fs[2][f],\
                    enc_fs[3][f], enc_fs[4][f], enc_fs[5][f], enc_fs[6][f]),1)
            
            feat_concat = self.cconv(enc_fs_concat)
            feat_fusion = self.ff(enc_fs_stack, feat_concat)
            out = feat_fusion

            efst_outs.append(out)

        return efst_outs

class FeatureInterpolation(torch.nn.Module):
    def __init__(self):
        super(FeatureInterpolation, self).__init__()
        
    def forward(self, im0, im1, flowt0, flowt1):

        # Warp input using backwrad warping
        wim0 = warp(im0,flowt0)
        wim1 = warp(im1,flowt1)

        # Simply blended warped images or features
        wimt = (wim0 + wim1)/2.

        return wimt


class STVUN(torch.nn.Module):
    def __init__(self):
        super(STVUN, self).__init__()

        self.nif = 64
        self.sf = 4

        self.encoder = Encoder(self.nif)
        self.efst = EFST(self.nif)
        self.decoder = Decoder(self.nif)
        self.fi = FeatureInterpolation()
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        self.pwcnet = PWCDCNet().cuda()
    
    def forward(self, ims, tList):

        # shape of ims : list of input images [[B,C,H,W], ...]
        # shape of tList : list of target time index (e.g. [1/4, 2/4, 3/4])

        b,c,h,w = ims[0].size()
    
        outs = torch.zeros([len(tList)+1, b,c, h*self.sf, w*self.sf]).cuda()
        

        # Get feature representation of each images
        enc_s = []
        for i in range((len(ims))):
            s = self.encoder(ims[i])
            enc_s.append(s)

        # Fuse or merge feautres using EFST
        enc_sf = self.efst(enc_s)
        

        # Spatial decoder
        dec_feat, rimg = self.decoder(enc_s[3], enc_sf) 
        rimg = F.pixel_shuffle(rimg, self.sf) 
        out = F.upsample(ims[3], scale_factor= self.sf, mode='bilinear') + rimg
        outs[0,:] = out

        # Flow estimator
        uI3 = F.upsample(ims[3], scale_factor=self.sf, mode='bilinear')
        uI4 = F.upsample(ims[4], scale_factor=self.sf, mode='bilinear')
        flow34 = self.pwcnet(uI3, uI4)
        flow43 = self.pwcnet(uI4, uI3)


        for l in range(len(tList)):

            featI = []
            
            t = tList[l]
            flowt0 = -t*(1-t)*flow34 + t*t*flow43
            flowt1 = (1-t)*(1-t)*flow34 -t*(1-t)*flow43
        
            # Feature interpolation network
            for i in range(len(enc_s[3])):
                fi = self.fi(enc_s[3][i], enc_s[4][i], flowt0, flowt1)                    
                featI.append(fi)


            # Generate LR intermediate frames
            dwI = (warp(ims[3], flowt0) + warp(ims[4], flowt1))/2.

            # Spatio-temporal decoder
            _, trimg = self.decoder(featI, dec_feat)
            trimg = F.pixel_shuffle(trimg, self.sf)

            wI = F.upsample(dwI, scale_factor=self.sf, mode='bilinear')
            out = wI + trimg
            outs[l+1, :] = out

        return outs
         
     
