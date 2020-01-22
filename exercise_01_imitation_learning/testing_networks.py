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

        self.classes = [[1., 0., 0.],  # turn left
                        [-1., 0.5, 0.],  # turn left + gas
                        [-1., 0., 0.8],  # turn left + brake
                        [1., 0., 0.],  # turn right
                        [1., 0.5, 0.],  # turn right + gas
                        [1., 0., 0.8],  # turn right + brake
                        [0., 0., 0.],  # do nothing
                        [0., 0.5, 0.],  # gas
                        [0., 0., 0.8]]  # brake

        self.features_2d = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 3, stride=1),
            torch.nn.LeakyReLU(negative_slope=0.2),  # 94x94
            torch.nn.Conv2d(2, 4, 3, stride=2),
            torch.nn.LeakyReLU(negative_slope=0.2),  # 46x46
            torch.nn.Conv2d(4, 8, 3, stride=2),
            torch.nn.LeakyReLU(negative_slope=0.2),  # 22x22
        ).to(gpu)
        self.scores = torch.nn.Sequential(
            torch.nn.Linear(3351, 64), # 8 * 22 * 22
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
        # conversion to gray scale
        observation = observation[:, :, :, 0] * 0.2989 + \
                      observation[:, :, :, 1] * 0.5870 + \
                      observation[:, :, :, 2] * 0.1140

        obs = observation.reshape(batch_size, 1, 96, 96)
        features_2d = self.features_2d(obs).reshape(batch_size, -1)
        return self.scores(features_2d)

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
        obs = observation[:, :84, :].reshape(batch_size, 1, 84, 96)

        # get features
        features_2d = self.features_2d(obs).reshape(batch_size, -1)
        #features_1d = self.features_1d(features_2d)

        fused_features = torch.cat((
            speed,  # batch_size x 1
            abs_sensors,  # batch_size x 4
            steering,  # batch_size x 1
            gyroscope,  # batch_size x 16
            features_2d), 1)
        #print(fused_features.shape)
        return self.scores(fused_features)

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
        # pass
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
        _, class_number = torch.max(scores[0], dim=0)
        steer, gas, brake = self.classes[class_number]
        return steer, gas, brake

class MultiClassNetwork(torch.nn.Module):
    def __init__(self):
        """
        The image size of the input is 96x96 pixels.
        """
        super().__init__()
        gpu = torch.device('cuda')
        # gpu = torch.device('cpu')

        self.features_2d = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 3, stride=1),
            torch.nn.BatchNorm2d(2),
            torch.nn.LeakyReLU(negative_slope=0.2),  # 94x94
            torch.nn.Conv2d(2, 4, 3, stride=2),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(negative_slope=0.2),  # 46x46
            torch.nn.Conv2d(4, 8, 3, stride=2),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(negative_slope=0.2)  # 22x22
        ).to(gpu)

        self.scores = torch.nn.Sequential(
            torch.nn.Linear(8 * 22 * 22, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(32, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(16, 8),
            torch.nn.BatchNorm1d(8),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(8, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Sigmoid()
        ).to(gpu)

    def forward(self, observation):
        batch_size = observation.shape[0]
        # conversion to gray scale
        observation = observation[:, :, :, 0] * 0.2989 + \
                      observation[:, :, :, 1] * 0.5870 + \
                      observation[:, :, :, 2] * 0.1140

        # extract sensor values
        #speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, batch_size)

        obs = observation.reshape(batch_size, 1, 96, 96)
        # crop and reshape observations to 84 x 96 to add sensor values
        #obs = observation[:, :84, :].reshape(batch_size, 1, 84, 96)

        features_2d = self.features_2d(obs).reshape(batch_size, -1)
        # features_1d = self.scores(features_2d)

        # fused_features = torch.cat((
        #     speed,  # batch_size x 1
        #     abs_sensors,  # batch_size x 4
        #     steering,  # batch_size x 1
        #     gyroscope,  # batch_size x 16
        #     features_2d), 1)
        # print(fused_features.shape)
        return self.scores(features_2d)

    def class_to_action(self, action_scores):
        steer_difference = action_scores[0][0] - action_scores[0][1]
        steer = -1.0 if steer_difference > 0.5 else 0.0
        steer = 1.0 if steer_difference < -0.5 else steer
        gas = 0.5 if action_scores[0][2] > 0.5 else 0.0
        brake = 0.8 if action_scores[0][3] > 0.5 else 0.0
        return steer, gas, brake

    def actions_to_classes(self, actions):

        return [torch.Tensor([int(action[0] < 0.), int(action[0] > 0.), int(action[1] > 0.), int(action[2] > 0.)]) for
                action in actions]


def extract_sensor_values(self, observation, batch_size):
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