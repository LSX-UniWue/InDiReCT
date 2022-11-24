from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class Explainer:
	def __init__(self, model, distance, transform, device="cpu", std_dev_spread=0.15, num_samples=25):
		self.model = model
		self.distance = distance
		self.device = device
		self.transform = transform

		self.base_embedding = self.generate_base_embedding()

		# SmoothGrad parameters
		self.std_dev_spread = std_dev_spread
		self.num_samples = num_samples
		
	def generate_base_embedding(self):
		# Creates a black image which is transformed/normalized
		base_img = self.transform(Image.fromarray(np.zeros((500, 500, 3)), mode="RGB")).unsqueeze(0).to(self.device)

		base_embedding = self.model(base_img).detach()

		return base_embedding

	def calc_grads(self, img):
		item = self.transform(img)
		data = item.to(self.device)
		
		img = item.detach()

		# add fake batch dimension
		data.unsqueeze_(0)
		data = data.repeat(self.num_samples, 1, 1, 1)
		data += torch.randn_like(data) * self.std_dev_spread * (data.max() - data.min())
		_ = data.requires_grad_()

		out = self.model(data)
		loss = self.distance(out, self.base_embedding).mean()
		loss.backward()

		grads = data.grad.mean(dim=1).detach().cpu()

		return img, grads

	@staticmethod
	def process_grads(grads, percentile=99):
		grads = grads.abs().sum(dim=0).detach().numpy()
		v_max = np.percentile(grads, percentile)
		v_min = np.min(grads)
		grads = np.clip((grads - v_min) / (v_max - v_min), 0, 1)

		return grads

	@staticmethod
	def visualize_grads(grads_processed, title="", save_path=None):
		plt.figure()
		plt.imshow(grads_processed, cmap="cividis")
		plt.axis('off')

		plt.title(title)

		if save_path:
			plt.savefig(save_path, bbox_inches="tight")

		plt.show()

	@staticmethod
	def visualize_img(img, title="", save_path=None):
		inv_normalize = transforms.Normalize(
			mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],
			std=[1/0.26862954, 1/0.26130258, 1/0.27577711]
		)

		img = inv_normalize(img)
		img_mod = img.permute(1, 2, 0).numpy()

		plt.figure()
		plt.imshow(img_mod)
		plt.axis('off')

		plt.title(title)

		if save_path:
			plt.savefig(save_path, bbox_inches="tight")

		plt.show()

