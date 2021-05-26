""" xvdp 
requires media-pipe and dlib

port from dlib to mediapipe for ffqh alightment - its faster
this does not use all of media pipe's data - 3d could be leveraged

install mediapipe
https://google.github.io/mediapipe/getting_started/install.html
or just pip install mediapipe
"""
import os
import os.path as osp
from urllib.parse import urlparse
import requests
from io import BytesIO


_DLIB = True
try:
    import dlib
except:
    _DLIB = False

import mediapipe as mp

import collections

import numpy as np
import scipy
from PIL import Image
import matplotlib.pyplot as plt


mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils 

##
# media pipe detection and segmetnation
#

def loadimg(url, crop=None):
    """ load PIL.Image from url
    """
    response = requests.get(url)
    img = None
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        if crop is not None:
            img = img.crop(crop)
    else:
        print("Code {} Failed to load <{}>".format(response.status_code, url))
    return img

def scrape(urls, outdir=None, crops=None, ext=".jpg"):
    """ scrape list of url images, returns PIl.Image
        optional: save image .jpg to folder
    Args
        urls    (list | str) valid url images
        outdir  (str [None]) if None, only saves files if path
        crop    (tuple (x0, y0, x1, y1) [None])
        ext     (str [.jpg]) save format
    """
    if isinstance(urls, str):
        urls = [urls]
    imgs = []
    for i, url in enumerate(urls):
        crop = None if crops is None else crops[i]
        img = loadimg(url, crop)
        if img is not None:
            imgs.append(img)
            if outdir is not None:
                os.makedirs(outdir, exist_ok=True)
                img.save(osp.join(outdir, url.split("=")[-1]+ext))
    return imgs
    
def _np_image(image, dtype='float32'):
    """  convert to np.array to dtype
    Args
        dtype   (str ['float32']) | 'uint8', 'float64 
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
        image   (path str | url str | np.ndarray | PIL.Image)
        as_np   (bool [False]) default: PIL.Image
        dtype   (str [None]) | 'uint8', |'float32', | 'float64'         
    """
    if isinstance(image, str):
        if osp.isfile(image):
            image = Image.open(image).convert('RGB')
        elif urlparse(image).scheme:
            image = loadimg(image)
        else:
            assert False, "invalid image {}".format(type(image))
    
    if isinstance(image, np.ndarray) and not as_np:
        if image.dtype != np.dtype('uint8'):
            image = (image*255).astype('uint8')
        image = Image.fromarray(image)
    elif as_np:
        image = _np_image(image, dtype=dtype)
    return image

def segment_face(image):
    """ returns image and multi_face_landmark list from mediapipe
    Args
        image   (str path | str url | PIL.Image | np.ndarray)
    """
    image = get_image(image, as_np=True)
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                                max_num_faces=2,
                                min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image)

    if not results.multi_face_landmarks:
          print("image {} contains no faces")
    return image, results.multi_face_landmarks

def detect_face(image):
    """returns image and detections list from mediapipe
    Args
        image   (str path | str url | PIL.Image | np.ndarray)
    """
    image = get_image(image, as_np=True)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image)
    return image, results.detections

##
# draw functions for media pipe, detection and segmentation
#
def mp_draw_segmentation(image, landmarks, thickness=2, circle_radius=2):
    """
    """
    drawing_spec = mp_drawing.DrawingSpec(thickness=thickness, circle_radius=circle_radius)
    annotated_image = image.copy()
    if landmarks:
        for face_landmarks in landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
    return annotated_image

def mp_draw_detection(image, detections):
    annotated_image = image.copy()
    if detections:
        for detection in detections:
            mp_drawing.draw_detection(annotated_image, detection)
    return annotated_image


def mp_land_array2d_pix(lmks, px, py):
    """ media pipe normalized landmarks to ndarray in pixel space
    """
    return mp_land_array2d(lmks)*np.asarray([px,py])

def mp_land_array2d(lmks):
    """ media pipe normalized landmarks to 2d normalizedndarray
    """
    return mp_land_array3d(lmks)[:,:2]

def mp_land_array3d(lmks, face_index=0):
    """ media pipe normalized landmarks to 3d normalizedndarray for first face only
    Args:
        lmks    (mediapipe face_mesh multi_face_landmarks)
    """
    out = []
    for nland in lmks.ListFields()[face_index][1]:
        out.append([nland.x, nland.y, nland.z])
    return np.asarray(out)

def mp_landmarks(image):
    """ returns 486 landmark array in pixel space
    https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/ 
    Args:
        image   (str)
    """
    image, landmarks = segment_face(image)
    if not landmarks:
        return None
    assert len(landmarks) == 1, "too many faces recognized, use multiface detector"

    py, px, _ = image.shape
    return mp_land_array2d_pix(landmarks[0], py=py, px=px)

##
# get landmarks from dlib
#
def dlib_landmarks(image, predictor_folder="."):
    """ run dlib to extract 68 face landmarks
    """
    assert _DLIB, "Cannot run dlib landmarks, install dlib first"

    _path = 'shape_predictor_68_face_landmarks.dat'
    _path = osp.join(osp.abspath(osp.expanduser(predictor_folder)), _path)
    if not osp.isfile(_path):
        _url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        _msg = "dlib 68 landmark file <{}> not found, wget {} and unzip or pass folder"
        assert osp.isfile(_path), _msg.format(_path, _url)

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(_path)

    dets = detector(image, 1)
    out = []
    for d in dets:
        shape = shape_predictor(image, d)
        for e in shape.parts():
            out.append([e.x, e.y])
    return np.asarray(out)

##
# align head with ffhq or dlib
#
def ffhq_align(img, landmarks=None,  output_size=1024, transform_size=4096, enable_padding=True, media_pipe=True):
    """ adapted from https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
        align image based on similarity transformation, eyes and mouth
        simplified version
        returns PIL Image.
        Args:
            img   PIL.Image object
            landmarks   np.ndarray
            media_pip   bool if True uses media  pipe landmarks

    media pipe face representation
    https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
    """
    img = get_image(img, as_np=False)
    if landmarks is None:
        landmarks = mp_landmarks(img)
        media_pipe = True

    # only eyes and mouth are used to align
    if not media_pipe:  # use dlib
        lm_eye_left      = landmarks[36 : 42]  # left-clockwise
        lm_eye_right     = landmarks[42 : 48]  # left-clockwise
        lm_mouth_outer   = landmarks[48 : 60]  # left-clockwise
    
    else: # media pipe landmarks are denser, use similar set as dlib for heuristics
        lipsUpperOuter = [61, 40, 37, 0, 267, 270, 291]
        lipsLowerOuter = [91, 84, 17, 314, 405, 375]
        leftEyeUpper0 = [387, 386, 385, 384]
        leftEyeLower0 = [263, 373, 380, 362]
        rightEyeUpper0 = [160, 159, 158, 157]
        rightEyeLower0 = [33, 144, 153, 133]

        # qurik of ffhq align, labeled opposite to character orientation
        lm_eye_right    = landmarks[leftEyeUpper0 + leftEyeLower0]
        lm_eye_left     = landmarks[rightEyeUpper0 + rightEyeLower0]
        lm_mouth_outer  = landmarks[lipsUpperOuter + lipsLowerOuter]  

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    print(crop)

    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)
    
    return img
