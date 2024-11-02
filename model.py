import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_PM(nn.Module):

    def __init__(self, n_map = (13 + 1), temporal = 5):
        super(LSTM_PM, self).__init__()

        self.n_map = n_map
        self.temporal = temporal

        self.convnet1_conv_1 = nn.Conv2d(3, 128, kernel_size = 9, padding = 4)
        self.convnet1_conv_2 = nn.Conv2d(128, 128, kernel_size = 9, padding = 4)
        self.convnet1_conv_3 = nn.Conv2d(128, 128, kernel_size = 9, padding = 4)
        self.convnet1_conv_4 = nn.Conv2d(128, 32, kernel_size = 5, padding = 2)
        self.convnet1_conv_5 = nn.Conv2d(32, 512, kernel_size = 9, padding = 4)
        self.convnet1_conv_6 = nn.Conv2d(512, 512, kernel_size = 1)
        self.convnet1_conv_7 = nn.Conv2d(512, self.n_map, kernel_size = 1)
        self.convnet1_pool_1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.convnet1_pool_2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.convnet1_pool_3 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.convnet2_conv_1 = nn.Conv2d(3, 128, kernel_size = 9, padding = 4)
        self.convnet2_conv_2 = nn.Conv2d(128, 128, kernel_size = 9, padding = 4)
        self.convnet2_conv_3 = nn.Conv2d(128, 128, kernel_size = 9, padding = 4)
        self.convnet2_conv_4 = nn.Conv2d(128, 32, kernel_size = 5, padding = 2)
        self.convnet2_pool_1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.convnet2_pool_2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.convnet2_pool_3 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.convnet3_conv_1 = nn.Conv2d(48, 128, kernel_size = 11, padding = 5)
        self.convnet3_conv_2 = nn.Conv2d(128, 128, kernel_size = 11, padding = 5)
        self.convnet3_conv_3 = nn.Conv2d(128, 128, kernel_size = 11, padding = 5)
        self.convnet3_conv_4 = nn.Conv2d(128, 128, kernel_size = 1, padding = 0)
        self.convnet3_conv_5 = nn.Conv2d(128, self.n_map, kernel_size = 1, padding = 0)

        self.lstm1_gx = nn.Conv2d(32 + 1 + self.n_map, 48, kernel_size = 3, padding = 1)
        self.lstm1_ix = nn.Conv2d(32 + 1 + self.n_map, 48, kernel_size = 3, padding = 1)
        self.lstm1_ox = nn.Conv2d(32 + 1 + self.n_map, 48, kernel_size = 3, padding = 1)

        self.lstm2_ix = nn.Conv2d(32 + 1 + self.n_map, 48, kernel_size = 3, padding = 1, bias = True)
        self.lstm2_ih = nn.Conv2d(48, 48, kernel_size = 3, padding = 1, bias = False)
        self.lstm2_fx = nn.Conv2d(32 + 1 + self.n_map, 48, kernel_size = 3, padding = 1, bias = True)
        self.lstm2_fh = nn.Conv2d(48, 48, kernel_size = 3, padding = 1, bias = False)
        self.lstm2_ox = nn.Conv2d(32 + 1 + self.n_map, 48, kernel_size = 3, padding = 1, bias = True)
        self.lstm2_oh = nn.Conv2d(48, 48, kernel_size = 3, padding = 1, bias = False)
        self.lstm2_gx = nn.Conv2d(32 + 1 + self.n_map, 48, kernel_size = 3, padding = 1, bias = True)
        self.lstm2_gh = nn.Conv2d(48, 48, kernel_size = 3, padding = 1, bias = False)

        self.central_map_pooling = nn.AvgPool2d(kernel_size = 9, stride = 8)


    def convnet1(self, image):
        """
        ConvNet 1: Initial feature encoder network
        Input: 
            Image -> 3 * 368 * 368
        Output: 
            Initial heatmap -> n_map * 45 * 45
        """
        x = self.convnet1_pool_1(F.relu(self.convnet1_conv_1(image)))
        x = self.convnet1_pool_2(F.relu(self.convnet1_conv_2(x)))
        x = self.convnet1_pool_3(F.relu(self.convnet1_conv_3(x)))
        x = F.relu(self.convnet1_conv_4(x))
        x = F.relu(self.convnet1_conv_5(x))
        x = F.relu(self.convnet1_conv_6(x))
        x = self.convnet1_conv_7(x)

        return x


    def convnet2(self, image):
        """
        ConvNet 2: Common feature encoder network
        Input: 
            Image -> 3 * 368 * 368
        Output: 
            features -> 32 * 45 * 45
        """
        x = self.convnet2_pool_1(F.relu(self.convnet2_conv_1(image)))
        x = self.convnet2_pool_2(F.relu(self.convnet2_conv_2(x)))
        x = self.convnet2_pool_3(F.relu(self.convnet2_conv_3(x)))
        x = F.relu(self.convnet2_conv_4(x))

        return x


    def convnet3(self, hide_t):
        """
        ConvNet 3: Prediction generator network
        Input: 
            Hidden state (t) -> 48 * 45 * 345
        Output:  
            Heatmap -> n_map * 45 * 45
        """
        x = F.relu(self.convnet3_conv_1(hide_t))
        x = F.relu(self.convnet3_conv_2(x))
        x = F.relu(self.convnet3_conv_3(x))
        x = F.relu(self.convnet3_conv_4(x))
        x = self.convnet3_conv_5(x)

        return x


    def lstm(self, x, hide_t_1, cell_t_1):
        """
        Common (conv) LSTM unit
        Inputs:
            X -> ( 32 + n_map +1 ) * 45 * 45
            Hidden state (t-1) -> 48 * 45 * 45
            Cell state (t-1) -> 48 * 45 * 45
        Outputs:
            Hidden state -> 48 * 45 * 45
            Cell state -> 48 * 45 * 45
        """
        # Input gate
        it = torch.sigmoid(self.lstm2_ix(x) + self.lstm2_ih(hide_t_1))
        # Forget gate
        ft = torch.sigmoid(self.lstm2_fx(x) + self.lstm2_fh(hide_t_1))
        # Output gate
        ot = torch.sigmoid(self.lstm2_ox(x) + self.lstm2_oh(hide_t_1))
        # g = c'
        gt = torch.tanh(self.lstm2_gx(x) + self.lstm2_gh(hide_t_1))

        cell = ft * cell_t_1 + it * gt
        hidden = ot * torch.tanh(cell)

        return cell, hidden


    def lstm0(self, x):
        """
        Initial (conv) LSTM unit
        Input:
            x - >( 32 + n_map +1 ) * 45 * 45
        Outputs:
            Hidden state -> 48 * 45 * 45
            Cell state -> 48 * 45 * 45
        """
        # Input gate
        ix = torch.sigmoid(self.lstm1_ix(x))
        # Output gate
        ox = torch.sigmoid(self.lstm1_ox(x))
        # g = c'
        gx = torch.tanh(self.lstm1_gx(x))
        # Because there is no C(t-1) in the initial LSTM, so no need to forget-gate

        cell = torch.tanh(gx * ix)
        hidden = ox * cell

        return cell, hidden


    def initial_stage(self, image, centralmap):
        """
        Initial stage
        Inputs :
            image - > 3 * 368 * 368
            central gaussian map -> 1 * 368 * 368
        Outputs :
            Initial heatmap -> n_map * 45 * 45
            Heatmap -> n_map * 45 * 45
            Hidden state -> 48 * 45 * 45
            Cell state -> 48 * 45 * 45
            New central gaussian map -> 1 * 45 * 45
        """
        initial_heatmap = self.convnet1(image)
        features = self.convnet2(image)
        centralmap = self.central_map_pooling(centralmap)

        x = torch.cat([initial_heatmap, features, centralmap], dim = 1)  # Lstm input in step t
        cell, hidden = self.lstm0(x)
        heatmap = self.convnet3(hidden)
        return initial_heatmap, heatmap, cell, hidden, centralmap


    def common_stage(self, image, centralmap, heatmap, cell_t_1, hide_t_1):
        """
        Common stage
        Inputs:
            Image - > 3 * 368 * 368
            Central gaussian map -> 1 * 45 * 45
            Heatmap -> n_map * 45 * 45
            Hidden state (t-1) -> 48 * 45 * 45
            Cell state (t-1) -> 48 * 45 * 45
        Outputs:
            new heatmap -> n_map * 45 * 45
            hidden state -> 48 * 45 * 45
            cell state -> 48 * 45 * 45
        """
        features = self.convnet2(image)

        x = torch.cat([heatmap, features, centralmap], dim = 1)  # Lstm input in step t
        cell, hidden = self.lstm(x, hide_t_1, cell_t_1)
        new_heat_map = self.convnet3(hidden)

        return new_heat_map, cell, hidden


    def forward(self, images, centralmap):
        """
        Common stage
        Inputs:
            images - >(temporal * channels) * w * h = (t * 3) * 368 * 368
            central gaussian map -> 1 * 368 * 368
        Outputs:
            heatmaps -> (T + 1)* n_map * 45 * 45 (+1 is for initial heat map)
        """
        heat_maps = []
        # Select the channels of the first frame of all the temporal sequences in the batch
        image = images[:, 0:3, :, :]
        # Generate heatmap and initial heatmaps of the first frames with passing them to the first (initial) stage
        initial_heatmap, heatmap, cell, hide, centralmap = self.initial_stage(image, centralmap)
        heat_maps.append(initial_heatmap)
        heat_maps.append(heatmap)
        # For the other frames in  temporal sequences, we generate heatmaps by passing them to the second (common) stage
        for i in range(1, self.temporal):
            image = images[:, (3 * i):(3 * i + 3), :, :]
            heatmap, cell, hide = self.common_stage(image, centralmap, heatmap, cell, hide)
            heat_maps.append(heatmap)

        return heat_maps

    def get_model(temporal, device):
        """
        Khởi tạo mô hình LSTM_PM và chuyển nó tới thiết bị được chỉ định

        Args:
            temporal (int): Số lượng khung hình cần xử lý trong chuỗi thời gian
            device (torch.device): Thiết bị để chạy mô hình (CPU hoặc GPU)

        Returns:
            LSTM_PM: Mô hình đã được khởi tạo
        """
        # Khởi tạo mô hình LSTM_PM với tham số temporal
        model = LSTM_PM(temporal=temporal)

        # Chuyển mô hình sang thiết bị được chỉ định
        model = model.to(device)

        return model