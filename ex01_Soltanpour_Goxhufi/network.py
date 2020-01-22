import torch


class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        gpu = torch.device('cuda')
        # gpu = torch.device('cpu')

        self.classes = [[1., 0., 0.],    # turn left
                        [-1., 0.5, 0.],  # turn left + gas
                        [-1., 0., 0.8],  # turn left + brake
                        [1., 0., 0.],    # turn right
                        [1., 0.5, 0.],   # turn right + gas
                        [1., 0., 0.8],   # turn right + brake
                        [0., 0., 0.],    # do nothing
                        [0., 0.5, 0.],   # gas
                        [0., 0., 0.8]]   # brake

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, stride=1),     # 94x94
            #torch.nn.MaxPool2d(2, 2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(4, 8, 3, stride=1),     # 92x92
            torch.nn.MaxPool2d(2, 2),               # 46x46
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(8, 16, 3, stride=1),    # 44x44
            torch.nn.MaxPool2d(2, 2),               # 22x22
            torch.nn.LeakyReLU(negative_slope=0.2),

        ).to(gpu)
        self.fully_layers = torch.nn.Sequential(
            # TODO: change the initial input of the fully_layers to 16*XX*XX
            torch.nn.Linear(3351, 96),  # default was -> 16* 22 * 22 but after adding sensor values adjusted to 3351
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(96, 64),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(16, 9),
            torch.nn.Softmax(dim=1)
        ).to(gpu)

    def forward_old(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        batch_size = observation.shape[0]
        # grayscaling
        observation = observation[:, :, :, 0] * 0.2989 + \
                      observation[:, :, :, 1] * 0.5870 + \
                      observation[:, :, :, 2] * 0.1140

        obs_gray = observation.reshape(batch_size, 1, 96, 96)
        conv_layers = self.conv_layers(obs_gray).reshape(batch_size, -1)
        return self.fully_layers(conv_layers)

    ########################## Adding more features to the input ###########################################
    def forward(self, observation):
        """
        1.2 a) Observations
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 84, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        # pass
        batch_size = observation.shape[0]

        # extract sensor values
        speed, abs_sensors, steering, gyroscope = extract_sensor_values(observation, batch_size)

        # conversion to gray scale
        observation = observation[:, :, :, 0] * 0.2989 + \
                      observation[:, :, :, 1] * 0.5870 + \
                      observation[:, :, :, 2] * 0.1140

        # crop and reshape observations to 84 x 96
        obs_gray = observation[:, :84, :].reshape(batch_size, 1, 84, 96)

        # get features
        conv_layers = self.conv_layers(obs_gray).reshape(batch_size, -1)
        #features_1d = self.features_1d(features_2d)

        combined_features = torch.cat((
            speed,
            abs_sensors,
            steering,
            gyroscope,
            conv_layers), 1)
        #print(combined_features.shape)
        return self.fully_layers(combined_features)

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """
        # class_tensor = []
        # for current_class in torch.Tensor(self.classes):
        #     for action in actions:

        return [torch.Tensor([int(torch.prod(action == this_class)) for this_class in torch.Tensor(self.classes)]) for
                action in actions]

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        _, class_nr = torch.max(scores[0], dim=0)
        steer, gas, brake = self.classes[class_nr]
        return steer, gas, brake

class MultiClassNetwork(torch.nn.Module):
    def __init__(self):
        """
        The image size of the input is 96x96 pixels.
        """
        super().__init__()
        gpu = torch.device('cuda')
        # gpu = torch.device('cpu')

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 3, stride=1),
            torch.nn.BatchNorm2d(2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(2, 4, 3, stride=1),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(4, 8, 3, stride=1),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(negative_slope=0.2),
            # torch.nn.Conv2d(8, 16, 3, stride=1),
            # torch.nn.MaxPool2d(2, 2),
            # torch.nn.BatchNorm2d(16),
            # torch.nn.LeakyReLU(negative_slope=0.2)
        ).to(gpu)

        self.fully_layers = torch.nn.Sequential(
            torch.nn.Linear(3351, 32),                  # 16x8x10 + 7 sensor values
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(negative_slope=0.2),
            # torch.nn.Linear(64, 32),
            # torch.nn.BatchNorm1d(32),
            # torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(32, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(16, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.LeakyReLU(negative_slope=0.2),
            # torch.nn.Linear(8, 4),
            # torch.nn.BatchNorm1d(4),
            # torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Sigmoid()
        ).to(gpu)

    def forward(self, observation):
        batch_size = observation.shape[0]

        # extract sensor values
        speed, abs_sensors, steering, gyroscope = extract_sensor_values(observation, batch_size)
        # conversion to gray scale
        observation = observation[:, :, :, 0] * 0.2989 + \
                      observation[:, :, :, 1] * 0.5870 + \
                      observation[:, :, :, 2] * 0.1140
        # obs_gray = observation.reshape(batch_size, 1, 96, 96)
        # crop and reshape observations to 84 x 96 to add sensor values
        obs_gray = observation[:, :84, :].reshape(batch_size, 1, 84, 96)
        #print(self.conv_layers(obs_gray).shape)
        conv_layers = self.conv_layers(obs_gray).reshape(batch_size, -1)

        combined_features = torch.cat((
            speed,
            abs_sensors,
            steering,
            gyroscope,
            conv_layers), 1)
        #print(conv_layers.shape)
        #print(combined_features.shape)
        return self.fully_layers(combined_features)

    def class_to_action(self, action_scores):
        # calculating the first two entries which declare the streerings, to combinate and get one steer action,
        # because we cant steer right + left at the same time: thresholds is at -+0.5
        difference = action_scores[0][0] - action_scores[0][1]
        #steer = 0
        if difference > 0.5:
            steer = -1.0
        elif difference < -0.5:
            steer = 1.0
        else:
            steer = 0
        if action_scores[0][2] > 0.5:
            gas = 0.5
        else:
            gas = 0.

        if action_scores[0][3] > 0.5:
            brake = 0.8
        else:
            brake = 0.0
        return steer, gas, brake

    def actions_to_classes(self, actions):

        return [torch.Tensor([int(action[0] < 0.), int(action[0] > 0.), int(action[1] > 0.), int(action[2] > 0.)]) for
                action in actions]


def extract_sensor_values(observation, batch_size):
    """
    observation:    python list of batch_size many torch.Tensors of size
                    (96, 96, 3)
    batch_size:     int
    return          torch.Tensors of size (batch_size, 1),
                    torch.Tensors of size (batch_size, 4),
                    torch.Tensors of size (batch_size, 1),
                    torch.Tensors of size (batch_size, 1)
    """
    speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
    speed = speed_crop.sum(dim=1, keepdim=True) / 255
    abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
    abs_sensors = abs_crop.sum(dim=1) / 255
    steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
    steering = steer_crop.sum(dim=1, keepdim=True)
    gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
    gyroscope = gyro_crop.sum(dim=1, keepdim=True)
    return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
