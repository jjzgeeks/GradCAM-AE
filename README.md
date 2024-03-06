# GradCAM-AE
These codes are about "GradCAM-AE: A New Shield Defense Against Poisoning Attacks on Federated Learning"

Here is a structure of FL_GradCAM:

![Image alt text.](/readme_pics/FL_GradCAM.png)
GradCAM-assisted defense against poisoning attacks on FL. The server arbitrarily selects an image (e.g., an image with the label \"bird\") from the global testing dataset to create GradCAM heat maps for every uploaded model update. These GradCAM heat maps flow into an autoencoder for malicious model detection.



![Image alt text.](/readme_pics/Autoencoder.png)
Autoencoder-based abnormal GradCAM heat map identification. The server flattens and concatenates GradCAM heat maps as input to the encoder
neural network, which compresses the GradCAM heat maps from a high dimension to a low dimension Z. The decoder neural network takes Z as its input
to reconstruct the original input GradCAM heat maps.



## Requirements
- Install requirements via  `pip install -r requirements.txt`


## How to run :point_down:
Enter into each folder and run the following command:
```
python FL_GradCAM_main.py 
```

## References
1. https://github.com/shyam671/Federated-Learning/tree/main/code/FedAvg



