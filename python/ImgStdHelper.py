# -*- encoding:utf-8 -*-
"""
    标准化图片格式
"""
import glob
import imghdr
import os
import PIL.Image
import ZCommonUtil
from PIL import ImageFile

__author__ = 'BBFamily'

ImageFile.LOAD_TRUNCATED_IMAGES = True


def std_img_from_root_dir(root_dir, a_ext):
    """
    找到某一个ext的图片所有，都转换正真实类型，不是jpeg的
    全部转换为jpeg
    Parameters
    ----------
    root_dir
    a_ext

    Returns
    -------

    """
    img_list = find_img_by_ext(a_ext, root_dir)
    all_ext = change_to_real_tyep(img_list)
    for ext in all_ext:
        if ext <> 'jpeg':
            if ext is None:
                ext = a_ext
            sub_img_list = find_img_by_ext(ext, root_dir)
            _ = map(lambda img: covert_to_jpeg(img), sub_img_list)
            change_to_real_tyep(sub_img_list)


def covert_to_jpeg(org_img, dst_img=None):
    im = PIL.Image.open(org_img)
    if dst_img is None:
        dst_img = org_img
    im.convert('RGB').save(dst_img, 'JPEG')


def find_img_by_ext(ext, root_dir):
    """
    只遍历根目录及一级子目录
    Parameters
    ----------
    ext
    root_dir

    Returns
    -------

    """
    dirs = [root_dir + name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    dirs.append(root_dir)
    img_list = list()
    for dr in dirs:
        sub_list = glob.glob(dr + "/*." + ext)
        img_list.extend(sub_list)
    return img_list


def change_to_real_tyep(img_list):
    all_type = []
    for img in img_list:
        if not ZCommonUtil.file_exist(img):
            continue
        real_type = imghdr.what(img)
        if all_type.count(real_type) == 0:
            all_type.append(real_type)
        if real_type is None:
            continue
        real_name = img[:img.rfind('.')] + '.' + real_type
        os.rename(img, real_name)
    return all_type
