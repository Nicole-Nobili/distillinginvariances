import torch
import numpy as np
import matplotlib.pyplot as plt

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
    labels_normal = torch.softmax(labels_normal, dim=1) #batch classes
    labels_shifted = torch.softmax(labels_shifted, dim=1) #batch classes
    a = labels_normal - labels_shifted
    return torch.sum(torch.norm(labels_normal - labels_shifted, dim=1))

def test_IM(loader, model):
    device = model.device
    directions = ["u", "d", "l", "r"]
    invariance_measures = []

    for images,labels in loader:
        #images: batch channel rows cols
        images = images.squeeze().to(device)
        shifted = []
        for img in images:
            np.random.shuffle(directions)
            sh = shift_preserving_shape(img, direction=directions[0],
                                                  max_shift=5).unsqueeze(0)
            if sh is not None:
                shifted.append(sh)
        shifted = torch.cat(shifted, dim=0)
        shifted = shifted.view(-1, shifted.shape[-1] * shifted.shape[-2])
        images = images.view(-1, images.shape[-1] * images.shape[-2])
        labels = model(images)
        shifted_labels = model(shifted)
        invariance_measures.append(invariance_measure(labels, shifted_labels).unsqueeze(0))
    return torch.sum(torch.cat(invariance_measures))
