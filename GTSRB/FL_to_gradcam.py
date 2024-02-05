import torch, os, cv2
import numpy as np
import matplotlib.pyplot as plt
#from torchcam.methods import GradCAM
# from torchcam.utils import overlay_mask
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet, resnet18, ResNet18_Weights, \
    resnet50, ResNet50_Weights,  regnet_y_800mf, RegNet_Y_800MF_Weights,\
    efficientnet_b4, EfficientNet_B4_Weights, efficientnet_b0, EfficientNet_B0_Weights,\
    vgg16, VGG16_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


# Define an Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# class MincosLoss(nn.Module):
#     def __init__(self):
#         super(MincosLoss, self).__init__()
#
#     def forward(self, input):
#         # pearson = torch.corrcoef(input)
#         # loss = torch.min(pearson)
#         # Normalize the rows
#
#         normalized_matrix = input / input.norm(dim=1, keepdim=True)
#         # Compute the pairwise cosine similarity
#         cosine_similarity = torch.mm(normalized_matrix, normalized_matrix.T)
#         loss = torch.min(cosine_similarity)
#         return loss
"""------------------------------------------------------------------------------------"""
""" The methods of finding out outliers"""
"""------------------------------------------------------------------------------------"""

###-----------------------------   Autoencoder   --------------------------------------
def find_outliers_autoencoder(gradcam_list, img, curr_round):
    img_b, img_h = gradcam_list[0].shape

    grad_cam_list = [gradcam_list[i].flatten() for i in range(len(gradcam_list))]
    cleaned_vector_list = [np.where(np.isnan(vec), 0, vec) for vec in grad_cam_list]
    matrix = np.vstack(cleaned_vector_list)
    row, input_size = matrix.shape #number users * 224 * 224
    # Convert vectors to PyTorch tensors
    vectors_tensor = torch.tensor(matrix, dtype=torch.float32)
    hidden_size = 128 # 64
    autoencoder = Autoencoder(input_size, hidden_size)
    # Define loss function and optimizer
    # criterion = MincosLoss()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)

    # Train the Autoencoder
    num_epochs = 200
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = autoencoder(vectors_tensor)
        #loss = - criterion(outputs)
        loss = criterion(outputs, vectors_tensor)
        loss.backward()
        optimizer.step()
    autoencoder.eval()
    with torch.no_grad():
        # Use the trained Autoencoder to detect malicious
        reconstructed_vectors = autoencoder(vectors_tensor)
        reconstruction_errors = torch.mean((reconstructed_vectors - vectors_tensor) ** 2, dim=1)

    num_clients = len(reconstructed_vectors)
     # save reconstructed gradcam maps
    restr_gradcams = [torch.reshape(reconstructed_vectors[id], (img_b, img_h)).detach().cpu().numpy() for id in range(num_clients)]
    interval = 6
    rounds_set = [list(range(1, 1 + interval)), list(range(49, 49 + interval)), list(range(94, 94 + interval-3))]
    p_rounds = list(np.concatenate(rounds_set))
    if curr_round in p_rounds:
        for idx in range(num_clients):
            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                                  restr_gradcams[idx],
                                                  use_rgb=True)
            heatmap = Image.fromarray(visualization)
            heatmap.save(f'./client_{idx}_{curr_round}_autoencoder.jpg')

    if curr_round == 1 or curr_round == 30 or curr_round == 100:
         print("Benign GradCAM maps of reconstruction:", torch.reshape(reconstructed_vectors[1], (img_b, img_h)))
         print("Malicious GradCAM maps of reconstruction:", torch.reshape(reconstructed_vectors[-1], (img_b, img_h)))

    #mean_cos = []
    #reconstructed_vectors = reconstructed_vectors.detach().cpu().numpy()
    #reconstructed_vectors = reconstructed_vectors.tolist()
    #for i, vector in enumerate(reconstructed_vectors):
    #    other_vectors = reconstructed_vectors[:i] + reconstructed_vectors[i + 1:]
    #    similarities = cosine_similarity([vector], other_vectors)[0]
    #    mean_cos.append(sum(similarities) / (num_clients  - 1))
    #print(mean_cos)
    # threshold = np.mean(mean_cos) - 1.0 * np.std(mean_cos)
    #threshold = np.mean(mean_cos) - 2.0 * np.std(mean_cos)
    #outlier_indices = np.where(mean_cos < threshold)[0]
    #outlier_indices = outlier_indices.tolist()


    """Method 1 threshold = mean + std"""
    # Set a threshold for anomaly detection (e.g., based on mean + 3*std)
    threshold = torch.mean(reconstruction_errors) + 1.5 * torch.std(reconstruction_errors)
    # Identify abnormal vectors
    outlier_indices = torch.where(reconstruction_errors > threshold)[0]
    outlier_indices = outlier_indices.tolist()

    """Method 2, just pick the top k without including the same vaule"""
    #outlier_indices = torch.topk(reconstruction_errors, k=2).indices
    #outlier_indices = outlier_indices.tolist()

    """Method 3 delete the same values"""
    # reconstruction_errors =  reconstruction_errors.tolist()
    # largest_errors = sorted(set(reconstruction_errors), reverse=True)[:2] # 2 is the number of malicious users
    # outlier_indices = [i for i, value in enumerate(reconstruction_errors) if value in largest_errors]

    reconstruction_errors = reconstruction_errors.cpu().numpy()
    threshold = threshold.cpu().numpy()

    print(reconstruction_errors)
    print("threshold:", threshold)



    loss = loss.detach().cpu().numpy()
    print(loss)
    return outlier_indices, reconstruction_errors, threshold



def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]
    return img



"""------------------------------------------------------------------------------------"""
""" Input all local weights"""
"""------------------------------------------------------------------------------------"""
""" 
input local weights that stored all local weight of client into a list,
output benign local weights and the index of malicious client
"""
def FL_to_Grad_CAM(local_weights, curr_round):
    # image processing method
    #mean, std = (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)
    img_path = "./6.ppm"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # Calculate mean and std for each channel
    img_t = transforms.ToTensor()(img)
    mean = torch.mean(img_t, dim=(1, 2))
    std = torch.std(img_t, dim=(1, 2))

    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)
                                         ])

    input_tensor = data_transform(img).unsqueeze(0)
    target_category = None
    gradcam_list = []
    gradcam_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    #gradcam_model = resnet18(weights=ResNet18_Weights.DEFAULT)

    idx = 0
    for local_weight in local_weights:
        gradcam_model.load_state_dict(local_weight)
        #target_layers = [gradcam_model.layer4[-1]]  #resnet18, resnet50
        # target_layers = [gradcam_model.features] # VGG16, efficientb0-7,
        # target_layers = [gradcam_model.trunk_output] # regnet_y_800mf
        target_layers = [gradcam_model.features[-1]] # mobilenet_v3_large
        #gradcam_model.load_state_dict(weight)

        """choose GradCAM"""
        gradcam = GradCAM(model=gradcam_model, target_layers=target_layers, use_cuda=True)
        grayscale_cam = gradcam(input_tensor=input_tensor, targets=target_category)
        grayscale_cam = grayscale_cam[0, :] # 加平滑

        """choose layerCAM"""
        # layercam = LayerCAM(model=gradcam_model, target_layers=target_layers, use_cuda=True)
        # grayscale_cam = layercam(input_tensor=input_tensor, targets=target_category)[0]  # 加平滑

        # """Use from torchcam.methods import GradCAM"""
        # # Choose a target layer for Grad-CAM
        # target_layer = gradcam_model.layer4[-1]  # Example: last layer of the ResNet's layer4
        # gradcam = GradCAM(gradcam_model, target_layer)
        # scores = gradcam_model(input_tensor)
        # grayscale_cam = gradcam(class_idx=7, scores=scores)[0]

        gradcam_list.append(grayscale_cam)
        interval = 6
        rounds_set = [list(range(1, 1+interval)), list(range(49, 49+interval)), list(range(94, 94+interval-3))]
        p_rounds = list(np.concatenate(rounds_set))
        if curr_round in p_rounds:
            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                              gradcam_list[idx],
                              use_rgb=True)
            heatmap = Image.fromarray(visualization)  # size=224x224
            heatmap.save(f'./client_{idx}_{curr_round}_visualization.jpg')
        idx += 1

    #outlier_indices = find_outliers_ECOD(big_matrix)  #identify malicious
    outlier_indices, recon_errors, threshold = find_outliers_autoencoder(gradcam_list, img, curr_round)  # identify malicious
    print("outlier_indices:", outlier_indices)


    out_idx = [18, 19]
    outlier_indices = out_idx

    init_index = [1] * len(gradcam_list)
    if len(outlier_indices) == 0:
        predict_labels = init_index  # predict all clients are benign.
    else:
        for i in outlier_indices:
            init_index[i] = 0  # 0 is malicious
        predict_labels = init_index
    return predict_labels,  recon_errors, threshold
