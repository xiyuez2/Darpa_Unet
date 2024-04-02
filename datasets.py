import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import PIL
import os 
import torch.nn.functional as F
import torchvision.transforms.functional as aug_f
import random
import cv2
import json 
from itertools import chain
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
import math
import copy
import flow_transforms
from h5image import H5Image
import glob
from random import choice, sample
import scipy

def generate_point(mask):
    # mask should be of shape h, w
    # sample a point
    if np.sum(mask) <= 0:
        return None
    
    while True:
        point = (np.random.randint(0, mask.shape[0]), np.random.randint(0, mask.shape[1]))
        
        if mask[point]:
            # print(point)
            return point

def generate_heatmap(mask):

    h,w = mask.shape
    point = generate_point(mask)
    if point is None:
        return np.zeros((h,w))
    x = np.arange(0, np.max((h,w)))
    y = np.arange(0, np.max((h,w)))
    X, Y = np.meshgrid(x, y)
    Z = ((X - point[1])**2 + (Y - point[0])**2)**0.5

    # Normalize Z to 0 - 1
    Z = Z / (h**2 + w**2)**0.5

    # # viz mask and Z and save them as png
    # plt.imshow(mask)
    # plt.savefig("mask.png")
    # plt.imshow(Z[:h,:w])
    # plt.savefig("Z.png")
    
    # raise

    return Z[:h,:w]

def setup_h5im(train = True):
    mapsh5i = {}
    exclude_map = set(['CO_SanchezRes','CA_NV_LasVegas','CO_VailE100K','CA_Cambria','ID_LakeWalcott','AZ_PioRico_Nogales'])
    map_files = sorted(glob.glob("/projects/bbym/shared/data/commonPatchData/256/15/*.hdf5"))
    if train:
        map_files = map_files[0:int(len(map_files)*0.8)]
    else:
        map_files = map_files[int(len(map_files)*0.8):]
    for file in map_files:
        mapname = os.path.basename(file).replace(".hdf5", "")
        if mapname not in exclude_map:
            mapsh5i[mapname] = H5Image(file, "r")
    print("Setup complete. Number of maps loaded:", len(mapsh5i))
    return mapsh5i


def unpatch_get_dice(items):
    print("lauch process")
    # vars, index = items
    preds, preds_max, legend_gt, legend_name, cut_shape, map_mask_im, shift_coord, viz, thre = items
    # thre = thre[index]
    print("fecthed vars")
    preds_thre = (preds & (preds_max > thre)).astype(np.int8)
    # preds = preds > 0.5
    final_dice = []
    print("start calcualte dice")
    
    for i in range(len(preds)):
        cur_legend_name = legend_name[i]
        print("for legend:", cur_legend_name)
        cur_gt = legend_gt[i]

        unpatch_predicted = unpatchify(preds_thre[i], (cut_shape[0], cut_shape[1], 1))
        pad_unpatch_predicted = np.pad(unpatch_predicted, [(shift_coord[0], shift_coord[1]), (shift_coord[2], shift_coord[3]), (0,0)], mode='constant')
        pad_unpatch_predicted = pad_unpatch_predicted.astype(int)
        
        # print("debugging", map.shape, pad_unpatch_predicted.shape)
        cur_masked_img = cv2.bitwise_and(pad_unpatch_predicted, map_mask_im) 
        # map_mask(map, pad_unpatch_predicted)

        intersection = np.logical_and(cur_masked_img.flatten(), cur_gt.flatten()).sum()
        gtsum = cur_gt.sum() 
        union = cur_masked_img.sum() + gtsum
        dice = (2.0 * intersection) / union
        print("dice:", dice, 'intersection:', intersection/gtsum, 'union:', union/gtsum)
        final_dice.append(dice)
        if viz:
            viz_seg = np.repeat(cur_masked_img[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            viz_gt = np.repeat(cur_gt[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            cv2.imwrite("eval_viz/"+cur_legend_name+"_pred.tif", viz_seg) 
            cv2.imwrite("eval_viz/"+cur_legend_name+"_gt.tif", viz_gt) 
            plt.figure(figsize=(20,20))
            plt.imshow(viz_seg[:,:,0])
            plt.savefig("eval_viz/"+cur_legend_name+"_pred.png")
            plt.figure(figsize=(20,20))
            plt.imshow(viz_gt[:,:,0])
            plt.savefig("eval_viz/"+cur_legend_name+"_gt.png")
        print("final dice:", np.mean(final_dice))
    return final_dice

def generate_map_patches(mapsh5i,percent_valid = 0.75):
    map_patches = []
    for map_name in mapsh5i.keys():
        patches = mapsh5i[map_name].get_patches(map_name) 
        cor = mapsh5i[map_name].get_map_corners(map_name) 

        valid_patches = [(map_name, layer, x, y) for layer in patches if layer.endswith('_poly') for x, y in patches[layer]]
        valid_layers = [key for key in patches.keys() if key.endswith('_poly')]
        random_patches = [(map_name,choice(valid_layers), 
                        sample(range(cor[0][0], cor[1][0]), 1)[0], 
                        sample(range(cor[0][1], cor[1][1]), 1)[0]) 
                        for _ in range(len(valid_patches))]

        n = len(valid_patches)
        map_patches += sample(valid_patches, int(n * percent_valid)) + sample(random_patches, int(n * (1 - percent_valid)))

    return map_patches

def sincolor(image, position):
    # image is of shape 3 H W
    pe = position[image.long()] # 3 H W n
    pe = pe.transpose(1,-1) # 3 n H W
    pe = pe.flatten(start_dim=0, end_dim=1) # 3n H W 
    return torch.cat([pe,image],dim = 0) # 3n+3 H W

def canny(image):
    # image is of shape 3 H W
    edges = cv2.Canny(image,100,200)
    return torch.cat([edges,image],dim = 0) # 4 H W

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def cut_map(image,patch_dims = (256,256),patch_overlap=32):
    # patch parameter
    patch_dims = (256,256)
    patch_overlap = 32
    patch_step = patch_dims[1]-patch_overlap
    # cut boundary of the map 
    shift_x = (image.shape[0]-patch_dims[0])%patch_step
    shift_y = (image.shape[1]-patch_dims[1])%patch_step
    shift_x_left = shift_x//2
    shift_x_right = shift_x - shift_x_left
    shift_y_left = shift_y//2
    shift_y_right = shift_y - shift_y_left
    # recoder the shift_coord
    shift_coord =  [shift_x_left, shift_x_right, shift_y_left, shift_y_right]
    
    map_im_cut = image[shift_x_left:image.shape[0]-shift_x_right, shift_y_left:image.shape[1]-shift_y_right,:]
    cut_shape = np.shape(map_im_cut)
    map_patchs = patchify(map_im_cut, (*patch_dims,3), patch_step)
    patch_shape = np.shape(map_patchs)
    map_patchs = map_patchs.reshape((-1,patch_dims[0],patch_dims[1],3))
    return map_patchs, shift_coord, cut_shape, patch_shape

class MAPData(data.Dataset):
    '''
    return:
        map_img: map image (3,resize_size,resize_size)
        legend_img: legend image (3,resize_size,resize_size)
        seg_img: segmentation image (3,resize_size,resize_size)
    '''
    def __init__(self, data_path=None,type="poly",range=None,resize_size = 256 , train = True, filp_rate = 0, color_jitter_rate = 0, edge = False, update_percent_valid_every = 1, heatmap = ''):
        print("augmentation rate:", filp_rate, color_jitter_rate)
        self.edge_only = edge
        self.heatmap = heatmap
        self.resize_size = resize_size
        self.train = train
        self.filp_rate = filp_rate
        self.color_jitter_rate = color_jitter_rate
        # if train:
        self.data_transforms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        # transforms.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue=0.5)
        ])
        # else:
        #     self.data_transforms = transforms.Compose([
        #     transforms.Resize((resize_size, resize_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225]),
        #     ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor()
        ])
        self.root = data_path
        self.type = type
        map_path = os.listdir(os.path.join(self.root,self.type,"map_patches"))
        legend_path = ['_'.join(x.split('_')[0:-2])+'.png' for x in map_path]
        
        if range is not None:
            map_path = map_path[range[0]:range[1]]
            legend_path = legend_path[range[0]:range[1]]
        
        self.map_path = [os.path.join(self.root,self.type,"map_patches",x) for x in map_path]
        self.legend_path = [os.path.join(self.root,self.type,"legend",x) for x in legend_path]
        self.seg_path = [os.path.join(self.root,self.type,"seg_patches",x) for x in map_path]
        self.pe = positionalencoding1d(4, 256)
        self.mapsh5i = setup_h5im(train = train)
        self.iterations = 0
        self.percent_valid = 1
        self.update_percent_valid_every = update_percent_valid_every
        if self.train:
            self.patches = generate_map_patches(self.mapsh5i, percent_valid = 1)
        else:
            print("warning set validation set percent_valid to 1.0")
            self.patches = generate_map_patches(self.mapsh5i, percent_valid = 1.)

    def update_percent_valid(self, percent_valid = 0.0):
        assert percent_valid <= 1 and percent_valid >= 0
        assert self.train
        print("warning: currently update percent valid is not doing anything, just keep valid percent to 1")
        self.percent_valid = 1.0 #percent_valid
        self.patches = generate_map_patches(self.mapsh5i, percent_valid)

    def __getitem__(self, index):
        # map_img = Image.open(self.map_path[index])
        map_name, layer_name, row, column = self.patches[index]
        map_img = self.mapsh5i[map_name].get_patch(row, column, map_name)
        legend_img = self.mapsh5i[map_name].get_legend(map_name, layer_name)
        seg_img = self.mapsh5i[map_name].get_patch(row, column, map_name, layer_name)
        
        map_img = Image.fromarray(np.uint8(map_img))
        legend_img = Image.fromarray(np.uint8(legend_img))
        seg_img = Image.fromarray(np.uint8(seg_img))

        # deep copy map_img to avoid change the original map
        raw_img = np.array(copy.deepcopy(map_img))
        super_pixel_img = cv2.bilateralFilter(raw_img, 5, 75, 75)
        super_pixel_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
        ])
        super_pixel_img = super_pixel_transform(super_pixel_img)

        # legend_img = Image.open(self.legend_path[index])
        # seg_img = Image.open(self.seg_path[index])
        seg_img = torch.tensor(np.array(seg_img).astype(float))
        seg_img = F.interpolate(seg_img.unsqueeze(0).unsqueeze(0), size=(self.resize_size,self.resize_size), mode='bilinear', align_corners=False).squeeze(0)
        seg_img = seg_img > 0.5
        seg_img = seg_img.type(torch.int32)
        # seg_img = seg_img.type(torch.float)
        # print("debug",np.shape(map_img))
        
        # edges = cv2.Canny(np.array(map_img),100,200)
        # get 'edge' by conv2d with kernel
        if self.edge_only:
            im = np.array(map_img)
            # cv2.imwrite(str(index)+"_map_patches.png", im)
            # print("debug for edge:",im.shape) # 256 256 3
            # pad im with 0
            # im = np.pad(im, ((1, 1), (1, 1), (0, 0)), mode='constant')
            # edges = im[1:-1,1:-1] * 8  - im[0:-2, 1:-1] - im[2:, 1:-1] - im[1:-1, 0:-2] - im[1:-1, 2:] - im[0:-2, 0:-2] - im[2:, 2:] - im[0:-2, 2:] - im[2:, 0:-2]
            kernel = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) 
            im = np.sum(im, axis = 2)
            edges = np.zeros_like(im)
            edges = scipy.signal.convolve2d(im,kernel)
            edges = np.absolute(edges)
            
            # save edges to png for debug
            # cv2.imwrite(str(index)+"_edges.png", (edges/np.max(edges)*255).astype(np.uint8) )
            edges = torch.from_numpy(edges[1:-1,1:-1].astype(np.float32)).unsqueeze(0) #  1 256 1256
            # edges = edges
        else:
            edges = None
        
        if self.train and random.random() < self.color_jitter_rate:
            hue_factor = float(torch.empty(1).uniform_(-0.5, 0.5))
            map_img = aug_f.adjust_hue(map_img, hue_factor)
            legend_img = aug_f.adjust_hue(legend_img, hue_factor)

        map_img = self.data_transforms(map_img)
        legend_img = self.data_transforms(legend_img)

        if self.edge_only:
            map_img = edges #torch.cat([edges,map_img],dim = 0)
            
        if self.train and random.random() < self.filp_rate:
            map_img = torch.flip(map_img, (-1,)) #map_img[:,:,::-1]
            legend_img = torch.flip(legend_img, (-1,)) #legend_img[:,:,::-1]
            seg_img = torch.flip(seg_img, (-1,)) #seg_img[:,:,:,::-1]

        if self.train and random.random() < self.filp_rate:
            # map_img = map_img[:,::-1,:]
            # legend_img = legend_img[:,::-1,:]
            # seg_img = seg_img[:,:,::-1,:]
            map_img = torch.flip(map_img, (-2,)) #map_img[:,:,::-1]
            legend_img = torch.flip(legend_img, (-2,)) #legend_img[:,:,::-1]
            seg_img = torch.flip(seg_img, (-2,)) #seg_img[:,:,:,::-1]
        
        # map_img = sincolor(map_img,self.pe)
        # legend_img = sincolor(legend_img,self.pe)
    
        if index == 0:
            print("debug in data loader")
            print("train:", self.train)
            print(map_img.size(),legend_img.size(),seg_img.size())
        
        self.iterations += 1
        if self.iterations % (len(self.patches) * self.update_percent_valid_every) == 0 and self.train and self.percent_valid > 0.1:
            percent_valid = self.percent_valid - 0.01
            print("update percent valid to: ", percent_valid)
            self.update_percent_valid(percent_valid)
        
        if len(self.heatmap) > 0:
            # print(seg_img.shape)
            point_map = seg_img[0].numpy() # H, W
            heat_map_img = generate_heatmap(point_map)
            legend_img = torch.cat([torch.tensor(heat_map_img).unsqueeze(0),legend_img],dim = 0).float()
        return {
            "map_img": map_img,    # 3 H W
            "legend_img": legend_img, # 1 3 H W - > 3 H W 
            "GT": seg_img[0], #.type(torch.LongTensor),    # 1 H W
            # "sup_msk": legend_seg, # 1 H W
            # "cls": 1,
            # 'qry_names': ["datasetid"+str(index)],
            'qry_ori_size': torch.tensor((256,256)),
            'superpixel': super_pixel_img
        }
    
    def __len__(self):
        return len(self.patches)

    def sample_tasks(self):
        pass
    def reset_sampler(self):
        pass

class eval_MAPData(data.Dataset):
    '''
    return:
        map_img: map image (3,resize_size,resize_size)
        legend_img: legend image (3,resize_size,resize_size)
        seg_img: segmentation image (3,resize_size,resize_size)
    '''
    def __init__(self, data_path=None, type="poly", range = None, resize_size = 256, mapName = '', viz = False, edge = False, legend = None):
        self.edge_only = edge
        self.resize_size = resize_size
        self.train = False # train
        self.viz = viz
        self.data_transforms = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        self.mask_transforms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor()
        ])
        self.root = data_path
        self.type = type 
        # read in map data:
        mapPath = os.path.join(data_path, mapName)
        jsonPath = os.path.join(data_path, mapName[0:-4]+'.json')
        # gtPath = data_path + "" + mapName[:-4] #_Xjgb_poly.tif
        gtPath = data_path + "/../validation_rasters/" + mapName[:-4] #_Xjgb_poly.tif
        print(gtPath)
        
        self.map = cv2.imread(mapPath)
        if self.viz:
            plt.figure(figsize=(20,20))
            plt.imshow(self.map)
            plt.savefig("eval_viz/"+mapName+"_map.png")
            plt.close()
        with open(jsonPath, 'r') as f:
            jsonData = json.load(f)
        
        # only keep polygon label
        # read in legend img
        legend_name = []
        legend_img = []
        legend_gt = []

        # print(jsonData)
        for label_dict in jsonData['shapes']:
            
            if not label_dict['label'].endswith('_poly'):
                continue
            if not (legend is None) and label_dict['label'] != legend:
                continue
            print(label_dict['label'],legend)
            
            legend_name.append(label_dict['label'])
            gt_file = gtPath + "_" + label_dict['label'] + ".tif"
            print(gt_file)
            gt_im = cv2.imread(gt_file)
            try:
                legend_gt.append(gt_im[:,:,0])
            except:
                print("No GT found, using all zero")
                legend_gt.append(np.zeros_like(self.map[:,:,0]))
            point_coord = label_dict['points']
            flatten_list = list(chain.from_iterable(point_coord))
            if point_coord[0][0] >= point_coord[1][0] or point_coord[0][1] >= point_coord[1][1] or (len(flatten_list)!=4):
                x_coord = [x[0] for x in point_coord]
                y_coord = [x[1] for x in point_coord]
                x_low, y_low, x_hi, y_hi = int(min(x_coord)), int(min(y_coord)), int(max(x_coord)), int(max(y_coord))

            else: x_low, y_low, x_hi, y_hi = [int(x) for x in flatten_list]
            legend_coor =  [(x_low, y_low), (x_hi, y_hi)]
            shift_pixel  = 4
            im_crop = self.map[y_low+shift_pixel:y_hi-shift_pixel, x_low+shift_pixel:x_hi-shift_pixel] # need to resize
            im_crop_resize = cv2.resize(im_crop, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
            legend_img.append(im_crop_resize.transpose(0,1,2))

        self.legend_name = legend_name # [:2]
        self.legend_img = legend_img # [:2]
        self.legend_gt = legend_gt # [:2]
        self.map_patches, self.shift_coord, self.cut_shape,self.patch_shape = cut_map(self.map)
        self.map_mask_im = map_mask(self.map)
        # print("debug: ", np.shape(legend_gt),np.min(legend_gt),np.max(legend_gt))

        # self.w =
        self.pe = positionalencoding1d(4, 256)
        print("when init dataset:", np.shape(legend_name), np.shape(legend_img), np.shape(legend_gt),np.shape(self.map_patches))
        

    def __getitem__(self, index):
        # return {}
        # map_img = Image.open(self.map_path[index])
        # legend_img = Image.open(self.legend_path[index])
        # seg_img = Image.open(self.seg_path[index])
        # map_shape = np.shape(self.map)
        patch_idx = index %  len(self.map_patches)
        
        legend_idx = index // len(self.map_patches)
        
        map_img = self.map_patches[patch_idx]
        # deep copy map_img to avoid change the original map
        raw_img = copy.deepcopy(map_img)
        super_pixel_img = cv2.bilateralFilter(raw_img, 5, 75, 75)
        super_pixel_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
        ])
        super_pixel_img = super_pixel_transform(super_pixel_img)

        legend_img = self.legend_img[legend_idx]
        seg_img = np.zeros((256,256)) #self.legend_gt[patch_idx,legend_idx]

        seg_img = torch.tensor(np.array(seg_img).astype(float))
        seg_img = F.interpolate(seg_img.unsqueeze(0).unsqueeze(0), size=(self.resize_size,self.resize_size), mode='bilinear', align_corners=False).squeeze(0)
        seg_img = seg_img > 0.5
        seg_img = seg_img.type(torch.int64)
        # seg_img = seg_img.type(torch.float)

        # seg_img = (seg_img > 0.0001).type(torch.float)
        # print("debug:", np.shape(map_img),np.shape(legend_img))
        if self.edge_only:
            im = np.array(map_img)
            # cv2.imwrite(str(index)+"_map_patches.png", im)
            # print("debug for edge:",im.shape) # 256 256 3
            # pad im with 0
            # im = np.pad(im, ((1, 1), (1, 1), (0, 0)), mode='constant')
            # edges = im[1:-1,1:-1] * 8  - im[0:-2, 1:-1] - im[2:, 1:-1] - im[1:-1, 0:-2] - im[1:-1, 2:] - im[0:-2, 0:-2] - im[2:, 2:] - im[0:-2, 2:] - im[2:, 0:-2]
            kernel = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) 
            im = np.sum(im, axis = 2)
            edges = np.zeros_like(im)
            edges = scipy.signal.convolve2d(im,kernel)
            edges = np.absolute(edges)
            
            # save edges to png for debug
            # cv2.imwrite(str(index)+"_edges.png", (edges/np.max(edges)*255).astype(np.uint8) )
            edges = torch.from_numpy(edges[1:-1,1:-1].astype(np.float32)).unsqueeze(0) #  1 256 1256
            # edges = edges
        else:
            edges = None

        map_img = self.data_transforms(Image.fromarray(map_img))
        legend_img = self.data_transforms(Image.fromarray(legend_img))# .unsqueeze(0)

        
        if self.train and random.random() < self.color_jitter_rate:
            hue_factor = float(torch.empty(1).uniform_(-0.5, 0.5))
            map_img = aug_f.adjust_hue(map_img, hue_factor)
            legend_img = aug_f.adjust_hue(legend_img, hue_factor)

        if self.edge_only:
            map_img = edges 

        legend_seg = seg_img.clone()
        legend_seg = legend_seg * 0 + 1

        # map_img = sincolor(map_img,self.pe)
        # legend_img = sincolor(legend_img,self.pe)

        # print("mean of msk",torch.mean(seg_img.type(torch.LongTensor).type(torch.float)))
        # print("debug:", map_img.size(),legend_img.size(),seg_img.size(),legend_seg.size())
        return {
            "map_img": map_img,    # 3 H W
            "legend_img": legend_img, # 1 3 H W -> 3 H W
            "GT": seg_img[0], #.type(torch.LongTensor),    # 1 H W
            # "sup_msk": legend_seg, # 1 H W
            # "cls": 1,
            'qry_names': ["datasetid"+str(index)],
            'ori_size': torch.tensor((256,256)),
            'superpixel': super_pixel_img,
        }

    def metrics(self,preds,thres = [0.5]):
        shape = np.shape(preds)
        preds = preds.reshape(len(self.legend_name),self.patch_shape[0],self.patch_shape[1],1,shape[-1],shape[-1],1)
        preds_max = np.array([np.max(preds, axis=0)]*len(self.legend_name)) - 0.00001
        preds =  preds > preds_max
        final_dice_thes = []
        print("done with pre")
        
        # from multiprocessing import Pool, Manager
        # manager = Manager()
        # shared_list = manager.list([preds, preds_max, self.legend_gt , self.legend_name, self.cut_shape, self.map_mask_im, self.shift_coord, False, thres]) 
        
        # print("launching pool")
        # with Pool(processes=1) as pool:
        #     p = pool.map(unpatch_get_dice,
        #                     [ (shared_list, i) for i in range(len(thres))] 
        #                 )
        # print(p)
        final_dice_thes = []
        for thre in thres:
            item = (preds, preds_max, self.legend_gt , self.legend_name, self.cut_shape, self.map_mask_im, self.shift_coord, self.viz, thre)
            curdice = unpatch_get_dice(item)
            final_dice_thes.append(curdice)
        print(final_dice_thes)
        return final_dice_thes

    def __len__(self):
        return len(self.map_patches) * len(self.legend_name)

    def sample_tasks(self):
        pass
    def reset_sampler(self):
        pass

def map_mask(imarray):
    gray = cv2.cvtColor(imarray, cv2.COLOR_BGR2GRAY)  # greyscale image
    # Detect Background Color
    pix_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    background_pix_value = np.argmax(pix_hist, axis=None)

    # Flood fill borders
    height, width = gray.shape[:2]
    corners = [[0,0],[0,height-1],[width-1, 0],[width-1, height-1]]
    for c in corners:
        cv2.floodFill(gray, None, (c[0],c[1]), 255)

    # AdaptiveThreshold to remove noise
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Edge Detection
    thresh_blur = cv2.GaussianBlur(thresh, (11, 11), 0)
    canny = cv2.Canny(thresh_blur, 0, 200)
    canny_dilate = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Keeping only the largest detected contour.
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    wid, hight = imarray.shape[0], imarray.shape[1]
    mask = np.zeros([wid, hight])
    mask = cv2.fillPoly(mask, pts=[contour], color=(1,1,1)).astype(int)
    
    
    return mask

