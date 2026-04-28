import matplotlib.colors as clr
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap
import numpy as np

def color_enumerate(iterable, start: int = 0, cmap: Colormap = get_cmap("inferno")):
    """
    same functionality as enumerate, but additionally yields sequential colors from
    a given cmap
    """

    n = start
    try:
        length = len(iterable)
    except TypeError:
        length = len(list(iterable))
    for item in iterable:
        yield n, cmap(n / (length - 1)), item
        n += 1

def colors_from_arr(a, cmap=None, start=0, end=1):
    if cmap is None:
        # https://davidjohnstone.net/lch-lab-colour-gradient-picker#fcf0ba,3774a0
        # orange -> purple
        # clist = ['#ff854d', '#ff834d', '#ff814e', '#ff7f4e', '#ff7d4f', '#ff7b4f', '#ff7950', '#ff7751', '#ff7551', '#ff7352', '#ff7152', '#fe6f53', '#fe6d54', '#fe6a55', '#fe6855', '#fd6656', '#fd6457', '#fd6258', '#fc6059', '#fc5e5a', '#fc5c5a', '#fb595b', '#fb575c', '#fa555d', '#fa535e', '#f9515f', '#f84f60', '#f84c61', '#f74a62', '#f64863', '#f64664', '#f54466', '#f44167', '#f33f68', '#f23d69', '#f23a6a', '#f1386b', '#f0366c', '#ef346d', '#ee316f', '#ec2f70', '#eb2c71', '#ea2a72', '#e92873', '#e82575', '#e62376', '#e52077', '#e41d78', '#e21b7a', '#e1187b', '#df157c', '#de127e', '#dc0f7f', '#db0c80', '#d90881', '#d70583', '#d60284', '#d40085', '#d20087', '#d00088', '#ce0089', '#cc008b', '#ca008c', '#c8008d', '#c6008e', '#c40090', '#c20091', '#bf0092', '#bd0094', '#bb0095', '#b80096', '#b60097', '#b30099', '#b0009a', '#ae009b', '#ab009c', '#a8009e', '#a5019f', '#a203a0', '#9f05a1', '#9c08a2', '#990aa3', '#960da5', '#930fa6', '#8f11a7', '#8c14a8', '#8815a9', '#8517aa', '#8119ab', '#7d1bac', '#791cad', '#751eae', '#701faf', '#6c21b0', '#6722b1', '#6223b2', '#5d25b2', '#5826b3', '#5227b4']
        # rose -> blue
        # clist = ['#ff7575', '#ff7477', '#ff7479', '#fe737a', '#fe727c', '#fe727e', '#fe7180', '#fd7182', '#fd7084', '#fc7086', '#fc6f87', '#fb6f89', '#fa6e8b', '#fa6e8d', '#f96e8f', '#f86e91', '#f76d92', '#f76d94', '#f66d96', '#f56d98', '#f46d9a', '#f36d9b', '#f26d9d', '#f16d9f', '#ef6da1', '#ee6da2', '#ed6da4', '#ec6da6', '#ea6da7', '#e96da9', '#e76dab', '#e66eac', '#e46eae', '#e36eb0', '#e16eb1', '#e06fb3', '#de6fb4', '#dc6fb6', '#da70b7', '#d970b9', '#d771ba', '#d571bc', '#d371bd', '#d172be', '#cf72c0', '#cd73c1', '#cb73c2', '#c974c4', '#c674c5', '#c475c6', '#c275c7', '#c076c8', '#bd76c9', '#bb77ca', '#b977cb', '#b678cc', '#b478cd', '#b179ce', '#af79cf', '#ac7ad0', '#a97bd1', '#a77bd2', '#a47cd2', '#a17cd3', '#9e7dd4', '#9c7dd4', '#997ed5', '#967ed6', '#937fd6', '#907fd7', '#8d80d7', '#8a80d7', '#8780d8', '#8481d8', '#8081d8', '#7d82d9', '#7a82d9', '#7783d9', '#7383d9', '#7083d9', '#6c84d9', '#6984d9', '#6585d9', '#6185d9', '#5e85d9', '#5a86d9', '#5686d9', '#5286d8', '#4d87d8', '#4987d8', '#4487d7', '#4087d7', '#3b88d6', '#3588d6', '#2f88d5', '#2988d5', '#2189d4', '#1789d4', '#0989d3']
        # beige -> blue
        clist = [
            "#fcf0ba",
            "#f9efb9",
            "#f6eeb8",
            "#f3edb7",
            "#f0ecb6",
            "#edebb5",
            "#eaeab4",
            "#e7eab3",
            "#e4e9b2",
            "#e1e8b1",
            "#dee7b0",
            "#dbe6b0",
            "#d8e5af",
            "#d5e4ae",
            "#d1e3ae",
            "#cee2ad",
            "#cbe1ad",
            "#c8e0ac",
            "#c5dfac",
            "#c2deab",
            "#bfddab",
            "#bcdcab",
            "#b9dbaa",
            "#b6daaa",
            "#b3d9aa",
            "#b0d8aa",
            "#add6a9",
            "#aad5a9",
            "#a7d4a9",
            "#a4d3a9",
            "#a1d2a9",
            "#9ed1a9",
            "#9bd0a9",
            "#98cfa9",
            "#95cea9",
            "#92cca9",
            "#8fcba9",
            "#8ccaa9",
            "#89c9a9",
            "#86c8a9",
            "#83c7a9",
            "#80c5aa",
            "#7dc4aa",
            "#7ac3aa",
            "#78c2aa",
            "#75c0aa",
            "#72bfaa",
            "#6fbeab",
            "#6cbdab",
            "#6abbab",
            "#67baab",
            "#64b9ab",
            "#61b8ac",
            "#5fb6ac",
            "#5cb5ac",
            "#5ab4ac",
            "#57b2ac",
            "#54b1ac",
            "#52b0ad",
            "#4faead",
            "#4dadad",
            "#4aabad",
            "#48aaad",
            "#46a9ad",
            "#44a7ad",
            "#41a6ad",
            "#3fa5ad",
            "#3da3ad",
            "#3ba2ad",
            "#39a0ad",
            "#379fad",
            "#369dad",
            "#349cad",
            "#329aad",
            "#3199ad",
            "#3098ad",
            "#2f96ad",
            "#2e95ac",
            "#2d93ac",
            "#2c92ac",
            "#2c90ac",
            "#2b8fab",
            "#2b8dab",
            "#2b8caa",
            "#2b8aaa",
            "#2b88aa",
            "#2b87a9",
            "#2c85a9",
            "#2d84a8",
            "#2d82a7",
            "#2e81a7",
            "#2f7fa6",
            "#307ea5",
            "#317ca4",
            "#327aa4",
            "#3379a3",
            "#3477a2",
            "#3676a1",
            "#3774a0",
        ]
        cmap = clr.LinearSegmentedColormap.from_list("my map", clist)
        start = 0.2
        end = 1
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    a = np.array(a)
    a = a.astype("float")
    a += np.abs(a.min())
    a -= a.min()
    a /= a.max()
    color_scale = start + a * (end - start)
    return cmap(start + a * (end - start))
