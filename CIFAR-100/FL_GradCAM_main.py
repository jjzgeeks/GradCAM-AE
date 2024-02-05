# -*- coding: utf-8 -*-
"""FadAvg
Automatically generated by Colaboratory.
"""

# Commented out IPython magic to ensure Python compatibility.
#   %load_ext tensorboard
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import random, math, csv
import time
import torch
import torch.nn as nn
import seaborn as sns
from PIL import Image
from torchvision.models import resnet, resnet18, ResNet18_Weights, regnet_y_800mf, RegNet_Y_800MF_Weights,\
    regnet_y_400mf, RegNet_Y_400MF_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.utils.data import Dataset, DataLoader
from FL_to_gradcam import FL_to_Grad_CAM
from torchvision import transforms, utils, datasets
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, matthews_corrcoef
from scipy.io import savemat
from collections import deque

torch.cuda.empty_cache()
# from torchsummary import summary
# set manual seed for reproducibility
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""## Partitioning the Data (IID and non-IID)"""



def iid_partition(dataset, clients):
    """
    I.I.D paritioning of data over clients
    Shuffle the data
    Split it between clients

    params:
      - dataset (torch.utils.Dataset): Dataset containing the Images
      - clients (int): Number of Clients to split the data between

    returns:
      - Dictionary of image indexes for each client
    """

    num_items_per_client = int(len(dataset) / clients)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]

    for i in range(clients):
        client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
        image_idxs = list(set(image_idxs) - client_dict[i])

    return client_dict


def non_iid_partition(dataset, n_nets, alpha):
    """
        :param dataset: dataset name
        :param n_nets: number of clients
        :param alpha: beta parameter of the Dirichlet distribution
        :return: dictionary containing the indexes for each client
    """
    y_train = np.array(dataset.targets)
    min_size = 0
    K = 100
    N = y_train.shape[0]
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = np.array(idx_batch[j])

    # net_dataidx_map is a dictionary of length #of clients: {key: int, value: [list of indexes mapping the data among the workers}
    # traindata_cls_counts is a dictionary of length #of clients, basically assesses how the different labels are distributed among
    # the client, counting the total number of examples per class in each client.
    return net_dataidx_map


"""## Federated Averaging

### Local Training (Client Update)

Local training for the model on client side
"""


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class ClientUpdate(object):
    def __init__(self, dataset, batchSize, learning_rate, epochs, idxs):
        self.train_loader = DataLoader(CustomDataset(dataset, idxs), batch_size=batchSize, shuffle=True)
        self.learning_rate = learning_rate
        self.epochs = epochs


    def train(self, model):
        criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.95, weight_decay = 5e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        e_loss = []
        for epoch in range(1, self.epochs + 1):

            train_loss = 0.0

            model.train()

            for data, labels in self.train_loader:
                if data.size()[0] < 2:
                    continue;

                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data)
                # calculate the loss
                loss = criterion(output, labels)
                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)
            # average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)
        total_loss = sum(e_loss) / len(e_loss)

        return model.state_dict(), total_loss


# Calculate the magnitude and degree of the given deep learning model
def mag_deg(u, v):
    pre_g = torch.cat([p.view(-1) for p in u.values()])
    pre_g = pre_g / torch.linalg.vector_norm(pre_g)
    pre_g = pre_g.detach().cpu().numpy()

    # curr_g = torch.cat([p.view(-1) for p in v.values()]).detach().cpu().numpy()
    curr_g = torch.cat([p.view(-1) for p in v.values()])
    curr_g = curr_g / torch.linalg.vector_norm(curr_g)
    curr_g = curr_g.detach().cpu().numpy()
    distance = np.linalg.norm(curr_g - pre_g)

    cos_similarity = np.dot(pre_g, curr_g) / (np.linalg.norm(pre_g) * np.linalg.norm(curr_g)) # np.linalg.norm(v) is the magnitude of v
    cos_similarity = np.clip(cos_similarity, -1.0, 1.0) # clip
    angle_rad = np.arccos(cos_similarity)
    # Convert the angle from radians to degrees
    angle_deg = np.degrees(angle_rad)
    return distance, angle_deg



def generate_malicious_noise(w, num_malicious, curr_round):
    eu_distance = []
    weights_avg = copy.deepcopy(w[0])
    for k in weights_avg.keys():
        for i in range(1, len(w)):
            weights_avg[k] += w[i][k]
        weights_avg[k] = torch.div(weights_avg[k], len(w))
    benign_global_models = copy.deepcopy(weights_avg)  # benign global model generated by attacker
    #malicious_local_models = copy.deepcopy(w[-num_malicious:]) #randomly select num_malicious users
    malicious_local_models = copy.deepcopy(random.sample(w, num_malicious))


    # target_layer_names = ['classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.3.weight', 'classifier.3.bias']
    # for state_dict in malicious_local_models:
    #     modified_state_dict = state_dict.copy()
    #     for layer_name in target_layer_names:
    #         if layer_name in modified_state_dict:
    #             layer_params = modified_state_dict[layer_name]
    #             # noise = torch.randn_like(layer_params) * 20
    #             # layer_params += noise  # Example: Add Gaussian noise
    #
    #             noise = torch.poisson(torch.ones_like(layer_params))
    #             layer_params += noise
    #     state_dict.update(modified_state_dict)
    #     eu_distance.append(mag_deg(state_dict, benign_global_models)[0])

    #add noise
    for state_dict in malicious_local_models:
        # std = random.sample(range(100, 200), k=1)[0]  # std has a large effect on the Euclidean distance
        # mean = random.sample(range(800, 1000), k=1)[0]  # mean has a large effect on the angle
        for key in state_dict:
            if 'weight' in key or 'bias' in key:
                #noise = torch.randn_like(state_dict[key]) * std + mean  # + 0  # 0.01 is std, 0 is mean
                #noise = torch.normal(noise)  # torch.poisson(rates)
                #noise = noise * torch.log(torch.tensor(curr_round + 1))
                #state_dict[key] = state_dict[key] + noise

                # noise = torch.poisson(torch.ones_like(state_dict[key])) / 50
                # state_dict[key] = state_dict[key] + noise

                # # uniform distribution
                #noise = torch.FloatTensor(state_dict[key].size()).uniform_(10.0, 20.0).to(device) * curr_round  # Adjust the range as needed
                noise = torch.FloatTensor(state_dict[key].size()).uniform_(1.0, 2.0).to(device)
                state_dict[key] += noise

        eu_distance.append(mag_deg(state_dict, benign_global_models)[0])
    upload_local_models = w + malicious_local_models
    return upload_local_models, eu_distance



"""### Server Side Training

Following Algorithm 1 from the paper
"""
def training(model, rounds, batch_size, lr, ds, data_dict, C, K, E, num_malicious, plt_title, plt_color, cifar_data_test,
             test_batch_size, criterion, num_classes, partition):
    """
    Function implements the Federated Averaging Algorithm from the FedAvg paper.
    Specifically, this function is used for the server side training and weight update

    Params:
      - model:           PyTorch model to train
      - rounds:          Number of communication rounds for the client update
      - batch_size:      Batch size for client update training
      - lr:              Learning rate used for client update training
      - ds:              Dataset used for training
      - data_dict:       Type of data partition used for training (IID or non-IID)
      - C:               Fraction of clients randomly chosen to perform computation on each round
      - K:               Total number of clients
      - E:               Number of training passes each client makes over its local dataset per round
      - tb_writer_name:  Directory name to save the tensorboard logs
    Returns:
      - model:           Trained model on the server
    """

    # measure time
    start = time.time()
    #buffer_size = 5 # define buffer, also FL sample interval
    buffer_size = 3
    buffer = deque(maxlen=buffer_size)
    pre_global_model = deque(maxlen=1)
    total_interval = []


    # training loss
    train_loss = []
    test_accuracy, best_accuracy_set, test_loss = [], [], []
    best_accuracy = 0

    TPR, FPR, cm_set, Acc_set, AUC, F1_set, MCC_set = [], [], [], [], [], [], []
    eu_distance_set = []
    malicious_idx_set, final_outlier_index_set = [], []
    g_dist, g_angle = [], []
    recsrt, thresholds = [], []
    

    for curr_round in range(1, rounds+1):
        print(f"Round {curr_round} starting... ")
        w, local_loss = [], []
        # Retrieve the number of clients participating in the current training
        m = max(int(C * K), 1)
        # Sample a subset of K clients according with the value defined before
        S_t = np.random.choice(range(K), m, replace=False)
        for k in S_t:
            # Compute a local update
            local_update = ClientUpdate(dataset=ds, batchSize=batch_size, learning_rate=lr, epochs=E,
                                            idxs=data_dict[k])
            # Update means retrieve the values of the network weights
            weights, loss = local_update.train(model=copy.deepcopy(model))

            w.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))

        attackers_loss = sum(local_loss) / len(local_loss)
        new_local_loss = local_loss + [attackers_loss for _ in range(num_malicious)]

        new_local_models, eu_distance = generate_malicious_noise(w, num_malicious, curr_round)
        eu_distance_set.append(eu_distance)

        real_label = [1 for i in range(len(new_local_models))]  # Initially, assume all users are benign, 1 as benign, 0 as malicious
        real_label[-num_malicious:] = [0 for _ in range(num_malicious)]

        if curr_round % buffer_size != 0:
            """input FL local weights of all clients to Grad-CAM, output predict labels"""
            predict_labels, recon_errors, threshold = FL_to_Grad_CAM(new_local_models, curr_round)  # elements in list is 0 or 1
            buffer.append(predict_labels)
            recsrt.append(recon_errors)
            thresholds.append(threshold)
            benign_idx_set = [index for index, value in enumerate(predict_labels) if value == 1]
        else:
            predict_labels,  recon_errors, threshold = FL_to_Grad_CAM(new_local_models, curr_round)
            buffer.append(predict_labels)
            recsrt.append(recon_errors)
            thresholds.append(threshold)
            # sum buffer by row
            total_interval.append(curr_round)
            buffer_array = np.array(buffer)
            stati_outlier = np.sum(buffer_array, axis=0)

            # y_pred = [0 if x <= math.ceil(buffer_size / 3) + 1 else 1 for x in stati_outlier]  # Classification results
            y_pred = [0 if x <= 1 else 1 for x in stati_outlier]
            final_outlier_index = [i for i, x in enumerate(y_pred) if x == 0]
            final_outlier_index_set.append(final_outlier_index)
            print(f"In the {curr_round}-th communication round, the predicted outliers are: {final_outlier_index}")
            benign_idx_set = [index for index, value in enumerate(y_pred) if value == 1]
            ## confusion matrix
            cm = confusion_matrix(real_label, y_pred)
            cm_set.append(cm)
            tp, fn, fp, tn = cm.ravel()
            # Compute True Positive Rate (TPR) and FPR, Accuracy
            tpr = tp / (tp + fn)
            TPR.append(tpr)
            print(f'TPR is: {tpr}')  # TPR

            fpr = fp / (fp + tn)
            FPR.append(fpr)

            Accuracy = (tp + tn) / (tp + fn + fp + tn)
            Acc_set.append(Accuracy)

            F1_score = (2 * tp) / (2 * tp + fp + fn)
            F1_set.append(F1_score)

            MCC = matthews_corrcoef(real_label, y_pred)
            MCC_set.append(MCC)

            # Plot confusion matrix
            class_labels = ['Malicious', 'Benign']
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
            # plt.title('Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.savefig('./{}-th_round_confusion_matrix.png'.format(curr_round))
            plt.close()

            # Compute Receiver Operating Characteristic (ROC) curve
            # fpr, tpr, thresholds = roc_curve(real_label, y_pred)
            # Compute Area Under the Curve (AUC) score
            auc = roc_auc_score(real_label, y_pred)
            AUC.append(auc)
            print(f'AUC: {auc}')

        benign_local_loss_set = [new_local_loss[i] for i in benign_idx_set]
        weights_avg = copy.deepcopy(new_local_models[benign_idx_set[0]])
        for k in weights_avg.keys():
            for i in benign_idx_set[1:]:
                new_local_models[i][k] = new_local_models[i][k].type(weights_avg[k].dtype)
                weights_avg[k] += new_local_models[i][k]
            weights_avg[k] = torch.div(weights_avg[k], len(benign_idx_set))
        global_weights = copy.deepcopy(weights_avg)
        # move the updated weights to our model state dict
        model.load_state_dict(global_weights)
        # loss
        loss_avg = sum(benign_local_loss_set) / len(benign_local_loss_set)
        train_loss.append(loss_avg)

        t_accuracy, t_loss = testing(model, cifar_data_test, test_batch_size, criterion, num_classes)
        test_accuracy.append(t_accuracy)
        test_loss.append(t_loss)

        if best_accuracy < t_accuracy:
            best_accuracy = t_accuracy
            best_accuracy_set.append(best_accuracy)
        print(curr_round, loss_avg, t_loss, t_accuracy, best_accuracy)

        """Calculate the distance and angle of two global models """
        if curr_round == 1:
            dist, angle = mag_deg(new_local_models[0], global_weights)
            g_dist.append(dist)
            g_angle.append(angle)
            pre_global_model.append(global_weights)
        else:
            dist, angle = mag_deg(pre_global_model[0], global_weights)
        g_dist.append(dist)
        g_angle.append(angle)
        print(f"Two global models distance is: {dist}, angle is: {angle}")


        # print("**********----------------------------------------------------------------------------------------**********")
        if curr_round == 200:
            lr = lr / 2
            E = E - 1

        if curr_round == 300:
            lr = lr / 2
            E = E - 2

    end = time.time()
    # save data as mat
    savemat("./FL_Gradcam_cifar100_results_{}_{}_{}_{}.mat".format(partition, rounds, m, num_malicious),
            {"TPR": TPR, "FPR": FPR,  "cm_set": cm_set, "train loss": train_loss, "test loss": test_loss, "test accuracy": test_accuracy, " best_accuracy_set":  best_accuracy_set,
             "AUC": AUC, "Accuracy": Acc_set,"F1_score": F1_set,"MCC_set": MCC_set, "Eu_distance": eu_distance_set,"g_distance": g_dist, "g_angle": g_angle,  "final_outlier_index_set":final_outlier_index_set,
             "reconstruction_errors": recsrt, "threshold": thresholds})

    # plot figures
    plt.figure()
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots()
    x_axis = np.arange(1, len(train_loss) + 1)
    y_axis1 = np.array(train_loss)
    y_axis2 = np.array(test_accuracy)
    y_axis3 = np.array(test_loss)

    ax.plot(x_axis, y_axis1, 'tab:' + 'green', label='train_loss')
    ax.plot(x_axis, y_axis2, 'tab:' + 'blue', label='test_accuracy')
    ax.plot(x_axis, y_axis3, 'tab:' + 'red', label='test_loss')
    ax.legend(loc='upper left')
    ax.set(xlabel='Number of Rounds', ylabel='Train Loss',
           title=plt_title)
    ax.grid()
    fig.savefig(plt_title+'.jpg', format='jpg')

    plt.figure()
    plt.plot(FPR, TPR)
    # naming the x axis
    plt.xlabel('False Positive Rate (FPR)')
    # naming the y axis
    plt.ylabel('True Positive Rate (TPR)')
    # giving a title to my graph
    plt.title('ROC curve')
    plt.savefig('./ROC_{}_{}_{}_{}.png'.format(partition, rounds, m, num_malicious))

    plt.figure()
    plt.plot(total_interval, TPR)
    # naming the x axis
    plt.xlabel('The t-th communication round')
    # naming the y axis
    plt.ylabel('True Positive Rate (TPR)')
    # giving a title to my graph
    plt.title('The accuracy of the server successfully detecting malicious clients')
    plt.savefig('./TPR_{}_{}_{}_{}.png'.format(partition, rounds, m, num_malicious))

    plt.figure()
    plt.plot(total_interval, MCC_set)
    plt.xlabel('The t-th communication round ')
    # naming the y axis
    plt.ylabel('MCC')
    plt.savefig('./MCC_{}_{}_{}_{}.png'.format(partition, rounds, m, num_malicious))


    plt.figure()
    plt.plot(total_interval, AUC, label='AUC')
    plt.plot(total_interval, Acc_set, label='Accuracy')
    plt.plot(total_interval, F1_set, label='F1-score')
    # naming the x axis
    plt.xlabel('Number of Rounds')
    # naming the y axis
    plt.ylabel('AUC')
    # giving a title to my graph
    #plt.title('The AUC')
    plt.legend(loc='best')
    plt.savefig('./AUC_{}_{}_{}_{}.png'.format(partition,rounds, m, num_malicious))

    plt.figure()
    round_index = list(range(1, rounds+1))
    g_dist.pop(0)
    g_angle.pop(0)
    plt.subplot(211)
    plt.plot(round_index, g_dist)
    plt.ylabel('Euclidean distance')

    plt.subplot(212)
    plt.plot(round_index, g_angle)
    plt.ylabel('Angle')
    plt.xlabel('Round')
    plt.savefig('./Dist_Angle_{}_{}_{}_{}.png'.format(partition, rounds, m, num_malicious))

    plt.show()
    plt.close()
    print("Training Done!")
    print("Total time taken to Train: {}".format(end - start))
    return model



"""## ResNet50 Model (W & W/O GN)
"""

"""## Testing Loop"""
def testing(model, dataset, bs, criterion, num_classes):
    # test loss
    model.cuda()
    test_loss = 0.0
    correct_class = list(0. for i in range(num_classes))
    total_class = list(0. for i in range(num_classes))

    test_loader = DataLoader(dataset, batch_size=bs)
    l = len(test_loader)
    model.eval()
    for data, labels in test_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        output = model(data)
        loss = criterion(output, labels)
        test_loss += loss.item() * data.size(0)

        _, pred = torch.max(output, 1)

        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())

        # test accuracy for each object class
        #for i in range(len(labels)):
        for i in range(num_classes):
            label = labels.data[i]
            correct_class[label] += correct[i].item()
            total_class[label] += 1

    # avg test loss
    test_loss = test_loss / len(test_loader.dataset)

    return 100. * np.sum(correct_class) / np.sum(total_class), test_loss


def get_mean_std(dataloader):
    # Compute the mean and standard deviation of all pixels in the dataset
    mean, std = 0.0, 0.0
    total_images = 0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    mean /= total_images
    std /= total_images
    return mean, std





if __name__ == '__main__':
    # parser

    partition = 'non-iid'

    # create transforms
    # mean, std = (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784) #cifar10
    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) #cifar100
    transform = transforms.Compose([transforms.ToTensor()])
    cifar_data_train = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
    client_batch_size = 64

    # loader = torch.utils.data.DataLoader(cifar_data_train, batch_size=client_batch_size, shuffle=True)
    # mean, std = get_mean_std(loader)

    transforms_train = transforms.Compose([transforms.ToTensor(),
                                                 transforms.RandomCrop(32, padding=4, padding_mode='reflect'),  # crop
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.Normalize(mean, std)])  # ransforms.Normalize(*stats)
    transforms_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    data_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)
    data_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)

    classes = np.array(list(data_train.class_to_idx.values()))
    # print(classes)
    classes_test = np.array(list(data_test.class_to_idx.values()))
    num_classes = len(classes_test)
    criterion = nn.CrossEntropyLoss()

    # Hyperparameters
    num_commrounds = 100
    client_fraction = 1.0
    num_clients = 18
    num_malicious = 2
    num_epochs = 20 # 25, 20  # the number of local clients training epochs
    #client_batch_size = 64
    client_lr = 1e-4 #2e-5, 1e-4 # client learning rate
    alpha_partition = 0.5 #0.5

    if partition == 'iid':
        data_dict = iid_partition(data_train, num_clients)  # Uncomment for idd_partition
    else:
        data_dict = non_iid_partition(data_train, num_clients, float(alpha_partition))

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    #model = regnet_y_800mf(weights=RegNet_Y_800MF_Weights.DEFAULT)
    #model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    # model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Replace the last fully connected layer
    #cifar_cnn.fc = torch.nn.Linear(cifar_cnn.fc.in_features, num_classes)

    model.cuda()
    plot_str = partition + '_' + 'commrounds_' + str(num_commrounds) + '_clientfraction_' + str(
        client_fraction) + '_numclients_' + str(num_clients) + '_clientepochs_' + str(
        num_epochs) + '_clientbs_' + str(client_batch_size) + '_clientlr_' + str(client_lr)

    trained_model = training(model, num_commrounds, client_batch_size, client_lr, data_train, data_dict,
                             client_fraction, num_clients, num_epochs, num_malicious, plot_str,
                             'green', data_test, 128, criterion, num_classes, partition)

