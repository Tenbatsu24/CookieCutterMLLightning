import torch.nn as nn


class BaseCompactTransformer(nn.Module):

    def toggle_image_latent(self, image_latent: bool):
        """
        Toggle the image latent in the classifier.
        """
        self.classifier.image_latent = image_latent

    def toggle_patch_latent(self, patch_latent: bool):
        """
        Toggle the patch latent in the classifier.
        """
        self.classifier.patch_latent = patch_latent

    @property
    def patch_latent(self):
        """
        Get the patch latent in the classifier.
        """
        return self.classifier.patch_latent

    @property
    def image_latent(self):
        """
        Get the image latent in the classifier.
        """
        return self.classifier.image_latent

    def forward(self, x, **kwargs):
        x = self.tokenizer(x)
        return self.classifier(x, **kwargs)
