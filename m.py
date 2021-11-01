import matplotlib.pyplot as plt
import numpy as np
from numpy.core.shape_base import hstack
import torch
import PIL
import cv2
import os

import visualize
from run import estimate


def compute_mask(fwd_flow, bwd_flow):
    h, w = fwd_flow.shape[:2]
    mask = np.full((h, w), False)
    pts1 = []
    pts2 = []
    for r1 in range(h):
        for c1 in range(w):
            if fwd_flow[r1, c1, 0]**2 + fwd_flow[r1, c1, 1]**2 > 30:
                r2 = round(r1 + fwd_flow[r1, c1, 1])
                c2 = round(c1 + fwd_flow[r1, c1, 0])
                if 0 <= r2 < h and 0 <= c2 < w:
                    r1_ = r2 + bwd_flow[r2, c2, 1]
                    c1_ = c2 + bwd_flow[r2, c2, 0]
                    if (r1_ - r1)**2 + (c1_ - c1)**2 < 3:
                        mask[r1, c1] = True
                        pts1.append([c1, r1])
                        pts2.append([c2, r2])
    pts1 = np.array(pts1, dtype=float)
    pts2 = np.array(pts2, dtype=float)
    return mask, pts1, pts2


i = 3
j = 23
k = 26
fname = f'{i}_{j}_{k}.npz'
if os.path.isfile(fname):
    data = np.load(fname)
    pt1 = data['pt1']
    pt2 = data['pt2']
    F = data['F']
else:
    fname1 = f'images/microwaves/img{i:02}_{j:04}.png'
    fname2 = f'images/microwaves/img{i:02}_{k:04}.png'
    img1 = np.array(PIL.Image.open(fname1))[:, :, :3]
    img2 = np.array(PIL.Image.open(fname2))[:, :, :3]
    ten1 = torch.FloatTensor(np.ascontiguousarray(img1[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    ten2 = torch.FloatTensor(np.ascontiguousarray(img2[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

    flow12 = estimate(ten1, ten2).numpy().transpose([1, 2, 0])
    flow21 = estimate(ten2, ten1).numpy().transpose([1, 2, 0])

    mask1, pts1, pts2 = compute_mask(flow12, flow21)
    mask2 = compute_mask(flow21, flow12)[0]

    viz12 = visualize.flow_to_color(flow12)
    viz21 = visualize.flow_to_color(flow21)
    plt.subplot(3, 2, 1)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('img1')
    plt.subplot(3, 2, 2)
    plt.imshow(img2)
    plt.axis('off')
    plt.title('img2')
    plt.subplot(3, 2, 3)
    plt.imshow(viz12)
    plt.axis('off')
    plt.title('flow12')
    plt.subplot(3, 2, 4)
    plt.imshow(viz21)
    plt.axis('off')
    plt.title('flow21')
    plt.subplot(3, 2, 5)
    plt.imshow(mask1)
    plt.axis('off')
    plt.title('mask1')
    plt.subplot(3, 2, 6)
    plt.imshow(mask2)
    plt.axis('off')
    plt.title('mask2')
    plt.show()

    F, mask = cv2.findFundamentalMat(pts1, pts2)
    i = 0
    while not mask[i]:
        i += 1
    pt1 = pts1[i][:, None]
    pt2 = pts2[i][:, None]
    np.savez_compressed(fname, pt1=pt1, pt2=pt2, F=F)

K = np.load('cam.npy')
E = K.T @ F @ K
R1, R2, t = cv2.decomposeEssentialMat(E)
P1 = np.hstack((K, np.zeros((3, 1))))
for R in [R1, R2]:
    for t_ in [t, -t]:
        P2 = K @ np.hstack((R, t_))
        p1 = cv2.triangulatePoints(P1, P2, pt1, pt2)
        p1 /= p1[3, 0]
        p2 = np.hstack((R, t_)) @ p1
        # print(p1, p2)
        if p1[2, 0] > 0 and p2[2, 0] > 0:
            print(R)
            print(t_)
# print(R, t)
