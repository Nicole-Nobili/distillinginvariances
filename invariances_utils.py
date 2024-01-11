import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchmetrics
from torch.utils.data import DataLoader

def visualize_tensor_as_image(tensor):
    # Assuming the input tensor is a square matrix
    if len(tensor.shape) != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("Input tensor must be a square matrix.")

    plt.imshow(tensor, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

def shift_not_preserving_shape(image, direction : str, max_shift: int):
    img = torch.clone(image)
    shift = np.random.randint(low=1, high= max_shift+1)
    visualize_tensor_as_image(img)
    if direction == "u":
        img = torch.roll(img, -shift, 0)
        img[-shift:,:] = torch.full(img[-shift:,:].shape, -1)
    elif direction == "d":
        img = torch.roll(img, shift, 0)
        img[:shift,:] = torch.full(img[:shift,:].shape, -1)
    elif direction == "l":
        img = torch.roll(img, -shift, 1)
        img[:,-shift:] = torch.full(img[:,-shift:].shape, -1)
    elif direction == "r":
        img = torch.roll(img, shift, 1)
        img[:,:shift] = torch.full(img[:,:shift].shape, -1)
    else:
        raise ValueError("wrong value passed")
    visualize_tensor_as_image(img)
    return img

def shift_preserving_shape(image, direction : str, max_shift: int):
    initial_dir = direction
    img = torch.clone(image)
    shift = np.random.randint(low=1, high= max_shift+1)
    shift = max_shift
    #visualize_tensor_as_image(img)
    row_length = img.shape[1]
    col_length = img.shape[0]
    if direction == "u":
        while shift > 0 and torch.sum(img[:shift, :]) != -1 * shift * col_length:
            shift = shift - 1
        if shift == 0:
            if initial_dir == "d":
                print("Image could not be shifted.")
                return None
            direction = "d"
            shift = np.random.randint(low=1, high= max_shift+1)
        else:
            img = torch.roll(img, -shift, 0)
    elif direction == "d":
        while shift > 0 and torch.sum(img[-shift:, :]) != -1 * shift * col_length:
            shift = shift - 1
        if shift == 0:
            if initial_dir == "l":
                print("Image could not be shifted.")
                return None
            direction = "l"
            shift = np.random.randint(low=1, high= max_shift+1)
        else:
            img = torch.roll(img, shift, 0)
    elif direction == "l":
        while shift > 0 and torch.sum(img[:, :shift]) != -1 * row_length * shift:
            shift = shift - 1
        if shift == 0:
            if initial_dir == "r":
                print("Image could not be shifted")
                return None
            direction = "r"
            shift = np.random.randint(low=1, high= max_shift+1)
        else:
            img = torch.roll(img, -shift, 1)
    elif direction == "r":
        while shift > 0 and torch.sum(img[:, -shift:]) != -1 * row_length * shift:
            shift = shift - 1
        if shift == 0:
            if initial_dir == "u":
                print("Image could not be shifted")
                return None
            direction = "u"
            shift = np.random.randint(low=1, high= max_shift+1)
        else:
            img = torch.roll(img, shift, 1)
    else:
        raise ValueError("wrong value passed")
    #visualize_tensor_as_image(img)
    return img

def invariance_measure(labels_normal, labels_shifted):

    #normalize tensors
    #print(labels_normal[0:5])
    labels_normal = torch.softmax(labels_normal, dim=1) #batch classes
    #print(labels_normal[0:5])
    labels_shifted = torch.softmax(labels_shifted, dim=1) #batch classes
    #print((labels_normal-labels_shifted)[0])
    #print(torch.norm(labels_normal - labels_shifted, dim=1)[0])
    return torch.mean(torch.norm(labels_normal - labels_shifted, dim=1))

def test_IM(loader, model, cnn, device):
    cnn.eval()
    model.eval()
    directions = ["u", "d", "l", "r"]
    invariance_measures = []
    invariance_measure_cnn = []
    n = 0
    correct_normal = 0
    correct_shifted = 0
    correct_normal_cnn = 0
    correct_shifted_cnn = 0
    random_affine = transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))

    for images,labels in loader:
        #images: batch channel rows cols
        labels = labels.to(device)
        images = images.squeeze().to(device)
        shifted = []
        non_shifted = []
        for img in images:
            np.random.shuffle(directions)
            sh = shift_preserving_shape(img, direction=directions[0], max_shift=5)
            #sh = random_affine(img.unsqueeze(0))
            if sh is not None:
                n = n + 1
                #visualize_tensor_as_image(img.cpu())
                #visualize_tensor_as_image(sh.cpu())
                shifted.append(sh.unsqueeze(0))
                non_shifted.append(img.unsqueeze(0))
        shifted = torch.cat(shifted, dim=0)
        non_shifted = torch.cat(non_shifted, dim=0)
        with torch.no_grad():
            cnn_unsh = cnn(non_shifted.unsqueeze(1))
            cnn_sh = cnn(shifted.unsqueeze(1))
            shifted = shifted.view(-1, shifted.shape[-1] * shifted.shape[-2])
            non_shifted = non_shifted.view(-1, non_shifted.shape[-1] * non_shifted.shape[-2])
            unshifted_labels = model(non_shifted)
            shifted_labels = model(shifted)
        correct_normal = correct_normal + torch.sum(torch.max(unshifted_labels, dim = 1)[1] == labels).item()
        correct_shifted = correct_shifted + torch.sum(torch.max(shifted_labels, dim= 1)[1] == labels).item()
        correct_normal_cnn = correct_normal_cnn + torch.sum(torch.max(cnn_unsh, dim = 1)[1] == labels).item()
        correct_shifted_cnn = correct_shifted_cnn + torch.sum(torch.max(cnn_sh, dim= 1)[1] == labels).item()
        invariance_measure_cnn.append(invariance_measure(cnn_unsh, cnn_sh).unsqueeze(0))
        invariance_measures.append(invariance_measure(unshifted_labels, shifted_labels).unsqueeze(0))
    print(f"Correct normal: {correct_normal/n}\n"
          + f"Correct shifted: {correct_shifted/n}\n"
          + f"Correct cnn normal: {correct_normal_cnn/n}\n"
          + f"Correct cnn shifted: {correct_shifted_cnn/n}\n")
    
    plt.subplot(1,2,1)
    shifted_image = transforms.ToPILImage()(sh)
    plt.imshow(shifted_image, cmap="gray")
    plt.subplot(1,2,2)
    shifted_image = transforms.ToPILImage()(img)
    plt.imshow(shifted_image, cmap="gray")
    print(torch.mean(torch.cat(invariance_measure_cnn)))
    return torch.mean(torch.cat(invariance_measures))

def validate(model: torch.nn.Module, weights_file: str, valid_data: DataLoader, device: str, mlp: bool):
    """Run the model on the test data and save all relevant metrics to file."""
    model.load_state_dict(torch.load(weights_file))
    model.to(device)
    nll = torch.nn.NLLLoss().to(device)
    ece = torchmetrics.classification.MulticlassCalibrationError(num_classes=10)

    batch_accu_sum = 0
    batch_nlll_sum = 0
    batch_ecel_sum = 0
    batch_invl_sum = 0
    totnum_batches = 0
    for (x,y) in valid_data:
        x = x.to(device)
        y_true = y.to(device)
        if (mlp):
            y_pred = model(x.view(-1,784))
        else:
            y_pred = model(x)
        accu = torch.sum(y_pred.max(dim=1)[1] == y_true) / len(y_true)

        log_probs = torch.nn.LogSoftmax(dim=1)(y_pred)
        nll_loss = nll(log_probs, y_true)
        ece_loss = ece(y_pred, y_true)

        batch_accu_sum += accu
        batch_nlll_sum += nll_loss
        batch_ecel_sum += ece_loss
        totnum_batches += 1

    metrics = {
        "accu": (batch_accu_sum / totnum_batches).cpu().item(),
        "nlll": (batch_nlll_sum / totnum_batches).cpu().item(),
        "ecel": (batch_ecel_sum / totnum_batches).cpu().item(),
    }
    for key, value in metrics.items():
        print(f"{key}: {value:.8f}")
        print("")

    return metrics