import numpy as np
import os
import pandas as pd
import random
import scipy.io
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class PennDataset(Dataset):

    def __init__(self, video_paths, joint_poses_paths, n_sequences=1, temporal=5, n_joints=13, std=1):
        # Model input size
        self.height = 368
        self.width = 368
        # Position maps size
        self.pos_map_h = 45
        self.pos_map_w = 45
        # Length of temporal sequence
        self.temporal = temporal
        # The number of joints that we want to predict their position
        self.n_joints = n_joints
        # sigma of joint position maps
        self.std = std
        # Generate temporal sequences
        self.temporal_sequences = self.gen_temporal_seq(video_paths, joint_poses_paths, temporal, n_sequences)

    def gen_temporal_seq(self, video_paths, joint_poses_paths, temporal, n_sequences):

        temporal_sequences = []
        for i, path in enumerate(video_paths):

            frames = os.listdir(path)
            frames.sort()
            # Number of video frames
            n_frames = len(frames)

            # Ignore videos with frames less than the temporal seq length
            if n_frames < self.temporal:
                continue

            for _ in range(n_sequences):
                seq = []
                start_index = random.randint(0, n_frames - temporal)

                for k in range(start_index, (start_index + temporal)):
                    seq.append([os.path.join(path, frames[k]), k])

                temporal_sequences.append([seq, joint_poses_paths[i]])

        return temporal_sequences

    def __getitem__(self, item):

        # Load the frames (.jpg) of sequence
        frames = self.temporal_sequences[item][0]
        frames.sort()
        # Open the file containing video joints positions (.mat)
        joint_positions = scipy.io.loadmat(self.temporal_sequences[item][1])

        # (images = model input) shape : (t*3) * 368 * 368
        images = torch.zeros(self.temporal * 3, self.width, self.height)
        # (pos_maps = model output = ground truth) shape : t * 13+1 *  45 * 45
        pos_maps = torch.zeros(self.temporal, self.n_joints + 1, self.pos_map_w, self.pos_map_h)
        # max(h,w) where h and w are the height and width of the bounding box
        maxbbox_list = torch.zeros(self.temporal)

        for i in range(self.temporal):

            img_path = frames[i][0]
            img = Image.open(img_path)
            h, w, c = np.array(img).shape
            # Get the ratio between raw image size and target size
            # In the following we need these to correction joints position after image resizing
            ratio_x = self.width / float(w)
            ratio_y = self.height / float(h)
            # normalize image
            img_transformer = self.get_img_transformer(self.width, self.height)
            img = img_transformer(img)
            images[(i * 3): (i * 3 + 3), :, :] = img

            # Generate position maps
            frame_number = frames[i][1]
            joints_pos = [[i, j] for i, j in
                          zip(joint_positions['x'][frame_number], joint_positions['y'][frame_number])]  # shape = 13 * 2
            pos_map = self.gen_pos_maps(joints_pos,
                                        self.pos_map_w,
                                        self.pos_map_h,
                                        ratio_x,
                                        ratio_y)

            pos_maps[i, :, :, :] = torch.from_numpy(pos_map)

            if frame_number < len(joint_positions['bbox']):
                _, _, bbw, bbh = joint_positions['bbox'][frame_number]
                bbw = (bbw * ratio_x) / (self.width / self.pos_map_w)
                bbh = (bbh * ratio_y) / (self.height / self.pos_map_h)
                maxbbox_list[i] = max(bbw, bbh)
            else:
                maxbbox_list[i] = maxbbox_list[i - 1]

        # Generate the Gaussian heat map
        centermap = self.center_map(self.width / 2.0,
                                    self.height / 2.0,
                                    10,  # std
                                    self.width,
                                    self.height)

        centermap = torch.from_numpy(centermap)
        centermap = centermap.unsqueeze_(0)

        return images.float(), pos_maps.float(), centermap.float(), maxbbox_list, frames

    def center_map(self, x_center, y_center, sigma, w, h):
        """ Generate a Gaussian heatmap centered at (x_center, y_center) with given sigma """
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        d2 = (grid_x - x_center) ** 2 + (grid_y - y_center) ** 2
        exponent = d2 / (2.0 * sigma ** 2)
        return np.exp(-exponent)

    def guassian_heatmap(self, peak_x, peak_y, std, w, h):
        w = np.arange(w)
        h = np.arange(h)
        x_grid, y_grid = np.meshgrid(w, h)
        numerator = (pow(x_grid - peak_x, 2) + pow(y_grid - peak_y, 2))
        variance = pow(std, 2)
        return np.exp(-1 * (numerator / 2.0) / variance)

    def gen_pos_maps(self, joint_poses, pos_map_w, pos_map_h, ratio_x, ratio_y):
        """
        Generate P+1 (P joints plus one background channel with size (pos_map_w × pos_map_h)) pos maps
        """
        n_joints = len(joint_poses)
        pos_maps = np.zeros((n_joints + 1, pos_map_h, pos_map_w))
        for i in range(n_joints):

            pos = joint_poses[i]
            x = pos[0]
            y = pos[1]

            if (x == y) and (x < 2):  # joint is not present in the image ...
                map = np.zeros((pos_map_h, pos_map_w))
            else:
                # Modifying the joint position
                x = (x * ratio_x) / (self.width / pos_map_w)
                y = (y * ratio_y) / (self.height / pos_map_h)
                # generating a gaussian heatmap which centered on x and y (joint position)
                map = self.guassian_heatmap(x, y, self.std, pos_map_w, pos_map_h)

            pos_maps[i, :, :] = map

        background = np.zeros((pos_map_h, pos_map_w))
        for w in range(pos_map_w):
            for h in range(pos_map_h):
                max_value = max(pos_maps[:, w, h])
                background[w, h] = max(1 - max_value, 0)

        pos_maps[len(joint_poses), :, :] = background

        return pos_maps

    def get_img_transformer(self, width, height):
        return T.Compose([
            T.Resize((width, height), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.temporal_sequences)


def load_dataset(data_path):
    frames_dir = "/frames/"
    clips = os.listdir(data_path + frames_dir)
    random.shuffle(clips)

    train_frames = [data_path + frames_dir + str(x) for x in clips[:1258]]
    test_frames = [data_path + frames_dir + str(x) for x in clips[1258:]]

    train_labels = [(str(x) + '.mat').replace('frames', 'labels') for x in train_frames]
    test_labels = [(str(x) + '.mat').replace('frames', 'labels') for x in test_frames]

    print('-' * 40)
    print('Train set  - total number of videos =', len(train_labels))
    print('Test set - total number of videos = ', len(test_labels))

    return train_frames, train_labels, test_frames, test_labels


def get_data_loaders(train_frames, train_labels, val_frames, val_labels, train_bs, val_bs):
    train_data = PennDataset(train_frames, train_labels)
    val_data = PennDataset(val_frames, val_labels)

    print('-' * 40)
    print('Train samples ( sample = a sequence of frames) =', len(train_data))
    print('Validation samples =', len(train_data))

    train_dl = DataLoader(train_data, batch_size=train_bs, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=val_bs, shuffle=True)

    return train_dl, val_dl
