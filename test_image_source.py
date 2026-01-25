from camera_input.image_source import ImageSource

source = ImageSource("test_images")

for i in range(3):
    image, path = source.get_next_image()
    print(f"Loaded image {i+1}: {path}")
    print("Image shape:", image.shape)
