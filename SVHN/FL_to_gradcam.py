import torch, os
import numpy as np
from torchcam.utils import overlay_mask
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet, resnet18, ResNet18_Weights, \
    resnet50, ResNet50_Weights,  regnet_y_800mf, RegNet_Y_800MF_Weights,\
    efficientnet_b4, EfficientNet_B4_Weights, efficientnet_b0, EfficientNet_B0_Weights,\
    vgg16, VGG16_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, LayerCAM
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
"""------------------------------------------------------------------------------------"""
""" The methods of finding out outliers"""
"""------------------------------------------------------------------------------------"""
def find_outliers_min_mean_cos(gradcam_list):
    grad_cam_list = [gradcam_list[i].flatten() for i in range(len(gradcam_list))]
    cleaned_vector_list = [np.where(np.isnan(vec), 0, vec) for vec in grad_cam_list]
    mean_cos = []
    for i, vector in enumerate(cleaned_vector_list):
        other_vectors = cleaned_vector_list[:i] + cleaned_vector_list[i + 1:]
        similarities = cosine_similarity([vector], other_vectors)[0]
        mean_cos.append(sum(similarities) / (len(cleaned_vector_list) - 1))
    print(mean_cos)
    threshold = np.mean(mean_cos) - 1.0 * np.std(mean_cos)
    #threshold = np.mean(mean_cos) - 1.5 * np.std(mean_cos)
    outlier_indices = np.where(mean_cos < threshold)[0]
    outlier_indices = outlier_indices.tolist()
    return outlier_indices


"""------------------------------------------------------------------------------------"""
""" Input all local weights"""
"""------------------------------------------------------------------------------------"""
""" 
input local weights that stored all local weight of client into a list,
output benign local weights and the index of malicious client
"""
def FL_to_Grad_CAM(local_weights, curr_round):
    # image processing method
    #mean, std = (0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)
    img_path = "./7.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    img_t = transforms.ToTensor()(img)
    # Calculate mean and std for each channel
    mean = torch.mean(img_t, dim=(1, 2))
    std = torch.std(img_t, dim=(1, 2))
    data_transform = transforms.Compose([#transforms.Resize(224),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std)
                                         ])

    input_tensor = data_transform(img).unsqueeze(0)
    target_category = None
    gradcam_list = []
    #gradcam_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    gradcam_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    #gradcam_model.fc = torch.nn.Linear(gradcam_model.fc.in_features, 10)
    idx = 0
    for local_weight in local_weights:
        gradcam_model.load_state_dict(local_weight)
        target_layers = [gradcam_model.layer4[-1]]  #resnet18, resnet50
        # target_layers = [gradcam_model.features] # VGG16, efficientb0-7,
        # target_layers = [gradcam_model.trunk_output] # regnet_y_800mf
        #target_layers = [gradcam_model.features[-1]] # mobilenet_v3_large
        #gradcam_model.load_state_dict(weight)

        """choose GradCAM"""
        gradcam = GradCAM(model=gradcam_model, target_layers=target_layers, use_cuda=True)
        grayscale_cam = gradcam(input_tensor=input_tensor, targets=target_category)[0]

        """choose layerCAM"""
        # layercam = LayerCAM(model=gradcam_model, target_layers=target_layers, use_cuda=True)
        # grayscale_cam = layercam(input_tensor=input_tensor, targets=target_category)[0]  # 加平滑

        gradcam_list.append(grayscale_cam)
        interval = 6
        rounds_set = [list(range(1, 1+interval)), list(range(49, 49+interval)), list(range(94, 94+interval))]
        p_rounds = list(np.concatenate(rounds_set))
        if curr_round in p_rounds:
           visualization = overlay_mask(img, Image.fromarray(grayscale_cam), alpha=0.6)
           visualization.save(f'./client_{idx}_{curr_round}_visualization.jpg')
        idx += 1
 
    outlier_indices = find_outliers_min_mean_cos(gradcam_list)  # identify malicious
    print(outlier_indices)

    init_index = [1] * len(gradcam_list)
    if len(outlier_indices) == 0:
        predict_labels = init_index  # predict all clients are benign.
    else:
        for i in outlier_indices:
            init_index[i] = 0  # 0 is malicious
        predict_labels = init_index
    return predict_labels

