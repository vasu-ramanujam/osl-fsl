import torch.utils.data


class SearchDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform):
        """
        :param img_dir: the directory where the images are stored
        :param label_dir: the directory where the labels are stored
        :param transform: the albumentations transformation applied to image and
        label
        """
        self.img_dir, self.label_dir = img_dir, label_dir
        self.img_names = []
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = self._get_image(img_name)
        label = self._get_label(img_name)
        img, label = self._transform(img, label) #help
        return img, label, img_name

    def __len__(self):
        return len(self.img_names)

    def _get_image(self, img_name):
        img_path = f'{self.img_dir}/{img_name}'
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return img

    def _get_label(self, img_name):
        base = img_name.rsplit('.', 1)[0]
        label_dir = f'{self.label_dir}/{base}.npy'
        label = np.load(label_dir).astype(np.int8)
        return label

    def _transform(self, img, label):
        img = np.array(img)
        transformed = self.transform(image=img, mask=label) #image=, mask=label
        img = transformed['image']
        label = transformed['mask']
        return img, label
