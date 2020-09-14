import torch 
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from PIL import Image



def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    T = len(x)
    x = torch.stack(x, 1)
    
    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')

    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))

    outs = []

    for t in range(T):
        outs.append(x[:,t])

    return outs


def load_pretrained_weight(model, path):

    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

def pad_img(x, scale_factor=4):

    b,c,h,w = x.size()
    wt = (( w//(64*scale_factor)+1)*64*scale_factor -w)
    ht = (( h//(64*scale_factor)+1)*64*scale_factor -h)
    wp = np.int(wt/2)
    hp = np.int(ht/2)
    x = F.pad(x, pad=[wp,wp,hp,hp], mode='replicate')

    return x, wp, hp

def tensor_to_numpy(x, is_cuda = True):

    if is_cuda:
        x = x.cpu().data.numpy()
    x = np.transpose(x.astype('float32'), [0,2,3,1])
    x = np.clip((x+1.)/2., 0,1)
    
    return x

def numpy_to_tensor(x, is_cuda = True):
    
    transformations = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        
    x = transformations(x)
    x = x.unsqueeze(0)
    if is_cuda:
        x = x.float().cuda()
    else:
        x = x.float()
    
    return x

def num_to_filename(cnt = 1):

    MAX_LEN = 6
    str_cnt = str(cnt)
    
    file_str = '0'
    for c in range(MAX_LEN - len(str_cnt)):
        file_str += '0'

    file_str += str_cnt

    return file_str

def clip_and_numpy(x, wp, hp):

    if(wp !=0 and hp !=0):
        x = x[:,:,hp:-hp,wp:-wp]
    elif(wp==0 and hp!=0):
        x = x[:,:,hp:-hp,:]
    elif(wp!=0 and hp==0):
        x = x[:,:,:,wp:-wp]
            
    x = tensor_to_numpy(x)[0]   

    return x

def save_img(img, save_path):
    canvas = np.asarray(img)*255
    canvas = canvas.astype(np.uint8)
    canvas = Image.fromarray(canvas)
    canvas.save(save_path)
