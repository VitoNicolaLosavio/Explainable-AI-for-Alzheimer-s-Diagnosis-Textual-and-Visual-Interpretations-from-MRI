import os, warnings, torch, cv2
import numpy as np
from alzheimer_disease.src.models.gradcam import GradCAM
from alzheimer_disease.src.modules.preprocessing import get_transformations


def get_gradcam(
        example,
        model,
        saved_path,
        threshold
):
    """
	Computes Gradient-weighted Class Activation Mapping (Grad-CAM), and
	returns a 3D segmentation mask over the `image` brain shape and above a fixed `threshold`.
	Args:
		example (dict): a testing set example.
		model (torch.nn.Module): the model to be loaded.
		saved_path (str): folder path where to find the model dump.
		threshold (int | float): heat threshold above which draw the mask.
		plot_results (bool): whether to plot the visual results.
		alpha (int): transparency channel. Between 0 and 255.
	Returns:
		image (numpy.ndarray): the input 3D image.
		mask (numpy.ndarray): the related computed 3D segmentation mask.
		pred (int): model prediction.
	"""
    warnings.filterwarnings('ignore')
    try:
        device = 'cpu'
        model.load_state_dict(
            torch.load(os.path.join(saved_path, model.name + '_best.pth'),
                       map_location=torch.device(device))
        )
        model.to(device)
        _, eval_transform = get_transformations(model.in_size)
        x = eval_transform([example])
        x_input = [
            torch.unsqueeze(x[0]['image'], 0).to(device),
            torch.unsqueeze(x[0]['data'], 0).to(device)
        ]
        cam = GradCAM(nn_module=model, target_layers='output_layers.relu')
        heatmap = cam(x=x_input)
        model.eval()
        with torch.no_grad():
            pred = model(x_input)
        image = x[0]['image'].squeeze().detach().cpu().numpy()
        label = int(x[0]['label'].squeeze().detach().cpu().numpy())
        pred = int(pred.argmax(dim=1).squeeze().detach().cpu().numpy())
        heatmap = heatmap.squeeze().detach().cpu().numpy()
        mask = _get_heatmap_mask(image, heatmap, threshold)
        return image, mask, pred, label, heatmap
    except OSError as e:
        print('\n' + ''.join(['> ' for i in range(30)]))
        print('\nERROR: model dump for\033[95m ' + model.name + '\033[0m not found.\n')
        print(''.join(['> ' for i in range(30)]) + '\n')


def _get_heatmap_mask(image, heatmap, threshold):
    """
	Computes a 3D segmentation mask over the `image` brain shape,
	according to Grad-CAM `heatmap` and above a fixed `threshold`.
	Args:
		image (numpy.ndarray): the input 3D image.
		heatmap (numpy.ndarray): the Grad-CAM 3D heatmap.
		threshold (int): heat threshold above which draw the mask.
	Returns:
		mask (numpy.ndarray): the computed 3D segmentation mask.
	"""
    bg_mask = np.zeros(image.shape, dtype='uint8')
    heatmap_mask = np.zeros(image.shape, dtype='uint8')
    for z in range(image.shape[2]):
        # findind brain contours
        gray = cv2.normalize(image[:, :, z], np.zeros((image.shape[1], image.shape[0])), 0, 255, cv2.NORM_MINMAX,
                             cv2.CV_8U)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmented = np.zeros_like(gray)
        cv2.drawContours(segmented, contours, -1, (255), thickness=cv2.FILLED)
        # removing background
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                if segmented[y][x] == 0:
                    bg_mask[y][x][z] = 1
                else:
                    break
        for x in range(image.shape[1] - 1, -1, -1):
            for y in range(image.shape[0] - 1, -1, -1):
                if segmented[y][x] == 0:
                    bg_mask[y][x][z] = 1
                else:
                    break
    heatmap_mask[np.where((heatmap > threshold) & (bg_mask != 1))] = 1
    return heatmap_mask
