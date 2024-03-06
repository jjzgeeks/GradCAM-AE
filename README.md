# GradCAM-AE
These codes are about "GradCAM-AE: A New Shield Defense Against Poisoning Attacks on Federated Learning"

Here is a structure of FL_GradCAM:
![Image Alt text](/readme_pics/FL_GradCAM.png "GradCAM-assisted defense against poisoning attacks on FL. The server arbitrarily selects an image (e.g., an image with the label \"bird \") from the global testing dataset to create GradCAM heat maps for every uploaded model update. These GradCAM heat maps flow into an autoencoder for malicious
model detection.")


## Requirements
- Install requirements via  `pip install -r requirements.txt`


## How to run :point_down:
Enter into each folder and run the following command:
```
python FL_GradCAM_main.py 
```

## References
1. https://github.com/shyam671/Federated-Learning/tree/main/code/FedAvg



