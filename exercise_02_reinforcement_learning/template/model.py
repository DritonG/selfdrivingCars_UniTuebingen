import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """

        super().__init__()

        self.device = device 
        self.action_size = action_size

        # TODO: Create network
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=2),
            nn.Dropout2d(p=0.2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.Dropout2d(p=0.5),
            nn.LeakyReLU(negative_slope=0.2),
        ).to(self.device)

        # Number of Linear input depends on output of conv2d layers
        linear_input_size = self.feature_size()
        # print("CHECK:flattened output: ", linear_input_size)

        self.head = nn.Sequential(
            nn.Linear(linear_input_size, 256),  # 16x8x10 + 7 sensor values
        ).to(self.device)

        self.q_values = nn.Linear(256, self.action_size).to(self.device)

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, 1, 96, 96).to(self.device))).view(1, -1).size(1)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """
        # TODO: Forward pass through the network

         # should be a single frame for now, according to the exercise sheet
        batch_size = observation.shape[0]
        # print("CHECK: batch_size: ", batch_size)
        # extract sensor values
        # speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, batch_size)
        # conversion to gray scale
        print("CHECK: observation.shape: ", observation.shape)
        observation = observation[:, :, :, 0] * 0.2989 + \
                      observation[:, :, :, 1] * 0.5870 + \
                      observation[:, :, :, 2] * 0.1140

        observation = observation.reshape(batch_size, 1, 96, 96)
        # crop and reshape observations to 84 x 96 to add sensor values
        # observation = observation[:, :84, :].reshape(batch_size, 1, 84, 96)
        observation = torch.tensor(observation, device=self.device)
        # print("CHECK:observation.shape: ", observation.shape)
        features_2d = self.conv_net(observation).reshape(batch_size, -1)
        features_1d = self.head(features_2d)
        # combined_features = torch.cat((
        #     speed,
        #     abs_sensors,
        #     steering,
        #     gyroscope,
        #     features_1d), 1)
        # print("CHECK:combined.shape: ", features_1d.shape)
        return self.q_values(features_1d)

    def extract_sensor_values(self, observation, batch_size):
        """ Extract numeric sensor values from state pixels
        Parameters
        ----------
        observation: list
            python list of batch_size many torch.Tensors of size (96, 96, 3)
        batch_size: int
            size of the batch
        Returns
        ----------
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 4),
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 1)
            Extracted numerical values
        """
        print("CHECK: batch_size: ", batch_size)
        print("CHECK: observation: ", observation)

        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        print("CHECK: speed_crop.shape: ", speed_crop.shape)
        print("CHECK: speed_crop: ", speed_crop)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
