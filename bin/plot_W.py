import argparse

from matplotlib import pyplot as plt
plt.switch_backend('agg')

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

parser = argparse.ArgumentParser(description='Plot the W from a CRF model')
parser.add_argument('ckpt_path', default=None, metavar='CKPT_PATH', type=str,
                    help='Path to the ckpt file of a CRF model')


def main():
    args = parser.parse_args()
    ckpt = torch.load(args.ckpt_path, map_location={'cuda:6':'cuda:7'})
    W = ckpt['state_dict']['crf.W'].cpu().numpy()[0].reshape((3, 3, 3, 3))
    
    weight = [W[0, 0],W[0, 1],W[0, 2],W[1, 0],W[1, 1],W[1, 2],W[2, 0],W[2, 1],W[2, 2]]
    fig,axes = plt.subplots(nrows=3,ncols=3)
    i=0
    for ax in axes.flat:
        
        im = ax.imshow(weight[i], vmin=-1, vmax=1, cmap='seismic')
        i+=1
        
    fig.colorbar(im, ax=axes.ravel().tolist())
#    plt.subplot(331)
#    plt.imshow(W[0, 0], vmin=-1, vmax=1, cmap='seismic')
#    plt.subplot(332)
#    plt.imshow(W[0, 1], vmin=-1, vmax=1, cmap='seismic')
#    plt.subplot(333)
#    plt.imshow(W[0, 2], vmin=-1, vmax=1, cmap='seismic')
#
#    plt.subplot(334)
#    plt.imshow(W[1, 0], vmin=-1, vmax=1, cmap='seismic')
#    plt.subplot(335)
#    plt.imshow(W[1, 1], vmin=-1, vmax=1, cmap='seismic')
#    plt.subplot(336)
#    plt.imshow(W[1, 2], vmin=-1, vmax=1, cmap='seismic')
#
#    plt.subplot(337)
#    plt.imshow(W[2, 0], vmin=-1, vmax=1, cmap='seismic')
#    plt.subplot(338)
#    plt.imshow(W[2, 1], vmin=-1, vmax=1, cmap='seismic')
#    plt.subplot(339)
#    plt.imshow(W[2, 2], vmin=-1, vmax=1, cmap='seismic')
##
#    fig.colorbar(plt, ax=axes.ravel().tolist())

#    plt.subplots_adjust(right=0.8)
#    plt.colorbar()

#    plt.show()
    plt.savefig('/mnt/lustre/yuxian/Code/NCRF-master/CKPT_PATH/Camelyon16/crf/plot_W.png')


if __name__ == '__main__':
    main()
