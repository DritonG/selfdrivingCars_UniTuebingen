import random
import time

import torch
from imitations import load_imitations
from network import ClassificationNetwork, MultiClassNetwork
from torchvision import transforms

# imports for plotting
import matplotlib.pyplot as plt

def train(data_folder, trained_network_file):
    """
    Function for training the network.
    """
    infer_action = MultiClassNetwork()
#    infer_action = ClassificationNetwork()
    print(infer_action)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-2)
    observations, actions = load_imitations(data_folder)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    batches = [batch for batch in zip(observations, infer_action.actions_to_classes(actions))]
    gpu = torch.device('cuda')
    #gpu = torch.device('cpu')

    nr_epochs = 150
    batch_size = 128
    number_of_classes = 4  # needs to be changed: 9 for classificationNetwork() and 4 if MultiClassNetwork()
    start_time = time.time()
    loss_plot = []

    # trying to augment the data
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    # ])

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(gpu))
            batch_gt.append(batch[1].to(gpu))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                                         (-1, number_of_classes))

                batch_out = infer_action(batch_in)
                # we tried another cross entropy loss for the multiclass network,
                # but the results weren't better so we switched back
                #criterion = torch.nn.BCELoss(reduction='mean')
                #loss = criterion(batch_out, batch_gt)
                loss = cross_entropy_loss(batch_out, batch_gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss

                batch_in = []
                batch_gt = []
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))
        loss_plot.append(total_loss)
    torch.save(infer_action, trained_network_file)

    # plotting the loss/learning curve
    plt.figure()
    plt.title("training curve")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(loss_plot, label="entropy-loss")
    plt.legend(loc="best")
    plt.show()


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    batch_out:      torch.Tensor of size (batch_size, number_of_classes)
    batch_gt:       torch.Tensor of size (batch_size, number_of_classes)
    return          float
    """
    # pass
    epsilon = 0.000001
    loss = batch_gt * torch.log(batch_out + epsilon) + (1 - batch_gt) * torch.log(1 - batch_out + epsilon)
    return -torch.mean(torch.sum(loss, dim=1), dim=0)


# class CelebA(data.Dataset):
#     """
#     A class to perform data augmentation on our imitations
#     """
#     def __init__(self):
#           self.data_arr = ... # define the data-array (load from file)
#           self.labels = ... # define the labels
#
#           self.transform = transforms.Compse([
#               transforms.RandomCrop(20),
#               transforms.RandomHorizontalFlip(),
#               transforms.ToTensor()])
#
#     def __getitem(self, index):
#          np_arr = self.data_arr[index, :]
#          y = self.labels[index]
#
#          ## reshape np_arr to 28x28
#          np_arr = np_arr.reshape(28, 28)
#
#          ## convert to PIL-image
#          img = Image.fromarray((np_arr*255).astype('uint8'))
#
#          #apply the transformations and return tensors
#          return self.transform(img), torch.FloatTensor(y)