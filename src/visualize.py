import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import create_model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def compute(self):
        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze(0).squeeze(0).detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        max_val = cam.max()
        if max_val > 0:
            cam = cam / max_val
        return cam


class Visualizer:
    def __init__(self):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(base_path, 'models')
        self.images_dir = os.path.join(base_path, 'dataset', 'images')
        self.results_dir = os.path.join(base_path, 'results', 'gradcam')
        os.makedirs(self.results_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Loaded as lists so ensemble members can be visualized and averaged.
        self.loaded_models = {
            'baseline_cnn': self._load_model_members('baseline_cnn'),
            'cnn_lstm': self._load_model_members('cnn_lstm')
        }

    def _load_model_members(self, model_name):
        member_paths = sorted(glob.glob(os.path.join(self.models_dir, f"{model_name}_member_*.pth")))
        loaded = []

        if member_paths:
            for path in member_paths:
                model = create_model(model_name)
                model.load_state_dict(torch.load(path, map_location='cpu'))
                model.eval()
                loaded.append(model)
            print(f"Loaded {len(loaded)} {model_name} member checkpoints")
            return loaded

        legacy_path = os.path.join(self.models_dir, f"{model_name}.pth")
        if os.path.exists(legacy_path):
            model = create_model(model_name)
            model.load_state_dict(torch.load(legacy_path, map_location='cpu'))
            model.eval()
            loaded.append(model)
            print(f"Loaded legacy checkpoint for {model_name}")
            return loaded

        print(f"Warning: No checkpoints found for {model_name}")
        return loaded

    def _choose_test_images(self, n_samples=8):
        if not os.path.isdir(self.images_dir):
            return []
        all_images = sorted(
            [
                os.path.join(self.images_dir, f)
                for f in os.listdir(self.images_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
        )
        if not all_images:
            return []
        if len(all_images) <= n_samples:
            return all_images
        random.seed(42)
        return sorted(random.sample(all_images, n_samples))

    def _list_all_images(self):
        if not os.path.isdir(self.images_dir):
            return []
        return sorted(
            [
                os.path.join(self.images_dir, f)
                for f in os.listdir(self.images_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
        )

    def _build_lstm_sequence(self, current_img_path, sorted_images):
        idx = sorted_images.index(current_img_path)
        start_idx = max(0, idx - 4)
        seq_paths = sorted_images[start_idx:idx + 1]
        while len(seq_paths) < 5:
            seq_paths.insert(0, seq_paths[0])
        return seq_paths

    def _prepare_single_input(self, img_path):
        image = Image.open(img_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)
        return image, tensor

    def _prepare_lstm_input(self, img_path, sorted_images):
        seq_paths = self._build_lstm_sequence(img_path, sorted_images)
        frames = []
        for p in seq_paths:
            image = Image.open(p).convert('RGB')
            frames.append(self.transform(image))
        sequence_tensor = torch.stack(frames, dim=0).unsqueeze(0)
        return sequence_tensor

    def _compute_cam_for_model(self, model_name, model, img_path, sorted_images):
        target_layer = model.resnet.layer4
        grad_cam = GradCAM(model, target_layer)
        model.zero_grad()

        speed = torch.tensor([[0.3]], dtype=torch.float32)
        if model_name == 'baseline_cnn':
            _, image_tensor = self._prepare_single_input(img_path)
            output = model(image_tensor, speed)
        else:
            sequence_tensor = self._prepare_lstm_input(img_path, sorted_images)
            output = model(sequence_tensor, speed)

        target = output.sum()
        target.backward()
        cam = grad_cam.compute()
        grad_cam.remove_hooks()
        return cam

    def create_heatmap(self, mask, image):
        mask = cv2.resize(mask, (image.width, image.height))
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0
        image_array = np.float32(np.array(image)[:, :, ::-1]) / 255.0
        blended = (0.45 * image_array) + (0.55 * heatmap)
        blended = np.clip(blended, 0, 1)
        return blended[:, :, ::-1]

    def generate_visualizations(self):
        print("Generating GradCAM visualizations...")

        image_paths = self._choose_test_images(n_samples=8)
        if not image_paths:
            print(f"No test images found in {self.images_dir}")
            return

        sorted_images = self._list_all_images()
        if not sorted_images:
            print(f"No test images found in {self.images_dir}")
            return
        for img_path in image_paths:
            original_image = Image.open(img_path).convert('RGB')
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            for model_name, members in self.loaded_models.items():
                if not members:
                    continue

                cams = []
                for member in members:
                    cam = self._compute_cam_for_model(model_name, member, img_path, sorted_images)
                    cams.append(cam)

                mean_cam = np.mean(np.stack(cams, axis=0), axis=0)
                heatmap = self.create_heatmap(mean_cam, original_image)

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(original_image)
                plt.title('Original')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(mean_cam, cmap='jet')
                plt.title('GradCAM Mask')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(heatmap)
                plt.title('GradCAM Overlay')
                plt.axis('off')

                plt.suptitle(f"{model_name} | {os.path.basename(img_path)}")
                plt.tight_layout()

                save_path = os.path.join(self.results_dir, f"{base_name}_{model_name}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved {save_path}")

        print(f"Visualization complete! Results saved to {self.results_dir}")

    def create_side_by_side_comparison(self):
        print("Creating side-by-side comparisons...")
        files = os.listdir(self.results_dir)
        cnn_files = sorted([f for f in files if f.endswith('_baseline_cnn.png')])

        for cnn_file in cnn_files:
            base_name = cnn_file.replace('_baseline_cnn.png', '')
            lstm_file = f"{base_name}_cnn_lstm.png"
            if lstm_file not in files:
                continue

            cnn_path = os.path.join(self.results_dir, cnn_file)
            lstm_path = os.path.join(self.results_dir, lstm_file)
            cnn_img = Image.open(cnn_path)
            lstm_img = Image.open(lstm_path)

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cnn_img)
            plt.title('Baseline CNN')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(lstm_img)
            plt.title('CNN-LSTM')
            plt.axis('off')

            plt.suptitle(f"Attention Comparison | {base_name}")
            plt.tight_layout()

            save_path = os.path.join(self.results_dir, f"comparison_{base_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved comparison {save_path}")


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.generate_visualizations()
    visualizer.create_side_by_side_comparison()