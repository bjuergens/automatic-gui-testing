from torchvision import transforms


def vae_transformation_functions(img_size: int):
    transformation_functions = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    return transformation_functions
