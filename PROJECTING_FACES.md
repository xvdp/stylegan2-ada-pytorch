# Projecting Faces Mod (@xvdp) with [MEDIAPIPE](https://google.github.io/mediapipe/)
<br>

## Possible failure points

***This mod was NOT tested thoroughly, it likely contains loose ends** <br><br>
***StyleGAN2 may fail if CUDA and CUDNN are not properly installed, legacy of the original repository runtime compiles .cu files**
<br>
## Requires:
`pip install mediapipe` <br>
In order to project faces competently, eyes and mouth need to be aligned with a similarity transform. <br>
As mentioned on the [README](README.md), "image should be cropped and aligned similar to the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset)"<br><br>
FFHQ alignment process is a simple, yet necessary, heuristic that involves, [DLIB](http://dlib.net) 68 landmarks pretrained front face detector, using eyes and mouth, scales, rotates and crops image to a default size of 1024x1024.This mod is done with the purpose of simplifying and serializing the process. <br><br>
Projection core code was not modified in this mod, it can be read inside `x_projector.py project()` is an inverse fitting of the generator using VGG16 pretrained model features.

### replaces DLIB with MEDIAPIPE
* Why: Even when returning more information Mediapipe's face detector is faster. Its accuracy is possibly better for the eyes but not necessarily for the lips, this is not important as FFHQ's heuristics only require eyes' and mouth's center. <br>
* Mediapipe detector is based on 486 3D Points comprising the entire Face from the [tfjs-models](https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/). <br><br>

## Run
```python
import x_align_faces, x_projector
image = x_align_faces.ffhq_align(image)

pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
Generator, outputs = x_projector.mp_project(pkl, image)
proj_img = outputs["image"]
# new images can be run without reloading model
Generator, outputs = x_projector.mp_project(Generator, image)
```
## Files
`x_project_w_mediapipe.ipynb`: example code of projection from the wild 
<br>
`x_align_faces.py`:   contains mediapipe as well as simplified DLIB code is ported from FFHQ project, single command w


```python
    ffhq_align(img, landmarks=None,  output_size=1024, transform_size=4096, enable_padding=True, media_pipe=True)
    """output: rotated scaled cropped PIL Image
    Args:
        img         ndarray image | pil image | image path | image url
    Args optional:
        landmarks   2d ndarray [None], if None, runs mp_landmarks
        media_pipe  bool [True], if False uses dlib
        --other args ported from ffhq download file
    """

    landmarks = mp_landmarks(image)
    """output: 2d ndarray of 486 landmarks
    Args
        image    ndarray image | pil image | file path | file url
    """
    landmarks = dlib_landmarks(image, predictor_folder)
    """output: 2d ndarray of 68 landmarks
    Args
      image               file path
      predictor_folder    folder containing unzipped 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    """
```
`x_projector.py`:   mod of `projector.py` making loading and saving files optional
```python
    mp_project(network, img, seed=303, num_steps=100, device="cuda", outputs=None, outdir=None, out_name="")
    """ ouput: tuple (G, out)
            G       Generator nn.Module
            out     dict of requested outputs default ["image"]
    Args:
        network     path str | url str | Generator module
        img         path str | np image | pil image
        seed        int [303] # random seed from project.py
        num_steps   int [100] # project defaults to 1000, but in cases tested inversion process does not converge beyond 100 steps
        device      str ['cuda'] # this was not tested on cpu
        outputs     list [None], if None, then ['image'] | optional: 'video', 'npz'
        outdir      str output dir to save [None], if not none, save outputs to dir
        out_name    str "" optional suffix to saved filennames
    """

```