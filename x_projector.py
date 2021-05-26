""" xvdp 2021 modified for serialization
"""
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
import os.path as osp
from time import perf_counter

import click
import imageio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}         ', end="\r")

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
    logprint()
    return w_out.repeat([1, G.mapping.num_ws, 1])

#----------------------------------------------------------------------------

# @click.command()
# @click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
# @click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
# @click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
# @click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
# @click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_pil = Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())


def _np_image(image, dtype='float32'):
    """  convert to dtype
    """
    image = np.asarray(image)
    if dtype is not None and image.dtype != np.dtype(dtype):
        # convert to uint.
        if dtype == 'uint8':
            if np.log2(image.max()) !=1: # float images with range up to 2
                image = image*255
        elif image.dtype == np.dtype('uint8'):
            image = image/255
        image = image.astype(dtype)
    return image

def get_image(image, as_np=False, dtype=None):
    """ open image as PIL.Image or ndarray of dtype, or convert

    Args
        image   (str | np.ndarray | PIL.Image)
        as_np   (bool [False]) default: PIL.Image
        dtype   (str [None]) | 'uint8', |'float32', | 'float64'         
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    if isinstance(image, np.ndarray) and not as_np:
        if image.dtype != np.dtype('uint8'):
            image = (image*255).astype('uint8')
        image = Image.fromarray(image)
    elif as_np:
        image = _np_image(image, dtype=dtype)
    return image

def get_network(network, device="cuda"):
    """ opens network for inference from 
    """
    G = None
    if isinstance(network, str):
        print('Loading networks from "%s"...' % network)
        with dnnlib.util.open_url(network) as fp:
            G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)
    elif isinstance(network, torch.nn.Module):
        G = network.requires_grad_(False).to(device)
    return G

def _to_format(image, out_format=None):
    """ return recentered image
    Args
        img         (torch.tensor)
        out_format  (str [None]) | float32, uint8
                None        centered torch tensor
                'float32'   np.ndarray, centered, clamped
                'uint8'     np.ndarray, centered, clamted
    """
    # unnormalize
    image = image.clone().detach().cpu()
    if out_format is None:
        return image

    _norm = 255 if out_format[0] == 'u' else 1
    image.add_(1).mul_(_norm/2).clamp_(0, _norm)
    image = image.permute(0, 2, 3, 1)[0]
    if _norm == 255:
        image = image.to(torch.uint8)
    return image.numpy()

def _to_image(G, projected_w_steps, index=-1, out_format='float32'):
    """ returns syntehsized image
    Args 
        G                   (torch.nn.Module) Generator Model
        projected_w_steps   (tensor list) outputs
        index               (int [-1]) index of desired image
        out_format          (str ['float32'])
    """
    projected_w = projected_w_steps[index]
    out = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    return _to_format(out, out_format=out_format)

def mp_project(network, img, seed=303, num_steps=100, device="cuda", outputs=None, outdir=None, out_name=""):
    """
    Args
        network     str (trained pkl)| nn.Module Generator
        img         str (image name) | PIL.Image | np.ndarray
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    G = get_network(network, device=device)

    # Load target image.
    target_pil = get_image(img, as_np=False, dtype=None)
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    # does it need to keep the whole tensor in gpu why?
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    out = {}
    if outputs is None:
        outputs = ["image"]

    if isinstance(outputs, str):
        outputs = [outputs]
    elif isinstance(outputs, tuple):
        outputs = list(outputs)

    out_format = "float32" if outdir is None else 'uint8'
    for o in outputs:
        if o[0] == "i": # "image"
            out["image"] = _to_image(G, projected_w_steps, index=-1, out_format=out_format)
        elif o[0] == "v": # video
            out["video"] = [_to_image(G, projected_w_steps, index=i, out_format=out_format)
                            for i in range(len(projected_w_steps))]
        elif o[0] == "p": # projected_steps
            out["npz"] = projected_w_steps
        else:
            print("out: <%s> not recognized"%o)

    if outdir:
        os.makedirs(outdir, exist_ok=True)

        _i = 0
        _inc = ""
        if "image" in out:
            _name = osp.join(outdir, "proj{}{}.png".format(out_name, _inc))
            while osp.isfile(_name):
                _i += 1
                _inc = "_{:04}".format(_i)
                _name = osp.join(outdir, "proj{}{}.png".format(out_name, _inc))
            Image.fromarray(out["image"], 'RGB').save(_name)

            _name = osp.join(outdir, "target{}{}.png".format(out_name, _inc))
            target_pil.save(_name)

        if "video" in out:
            _name = osp.join(outdir, "proj{}{}.mp4".format(out_name, _inc))
            print (f'Saving optimization progress video {_name}')
            with imageio.get_writer(_name, mode='I', fps=10, codec='libx264', bitrate='16M') as video:
                for _frame in out["video"]:
                    video.append_data(np.concatenate([target_uint8, _frame], axis=1))
        if "npz" in out:
            _name = osp.join(outdir, "projected_w{}{}.npz".format(out_name, _inc))    
            np.savez(_name, w=projected_w_steps[-1].unsqueeze(0).cpu().numpy())

    return G, out
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
