from PIL import Image, ImageDraw, ImageFont
import numpy as np

def get_contrasting_color(image, x, y, width, height):
	"""
	Determine a contrasting color (black or white) based on the average color of a specified area in the image.
	"""
	# Crop the relevant part of the image
	cropped_image = image.crop((x, y, x + width, y + height))
	# Convert to numpy array for analysis
	np_image = np.array(cropped_image)
	# Calculate the average color
	average_color = np.mean(np_image, axis=(0, 1))
	# Brightness calculation based on perceived luminance
	brightness = np.sqrt(0.299 * average_color[0] ** 2 + 0.587 * average_color[1] ** 2 + 0.114 * average_color[2] ** 2)
	# Return white for dark backgrounds and black for light backgrounds
	return 'white' if brightness < 128 else 'black'

# def concatenate_image(images, rows, columns, separator_width=10):
# 	# Ensure we have the exact number of images needed
# 	if len(images) != rows * columns:
# 		raise ValueError(f"Expected {rows * columns} images, but got {len(images)}.")

# 	# Calculate the max width and height of images to standardize sizes
# 	max_width = max(img.width for img in images)
# 	max_height = max(img.height for img in images)

# 	# Resize images to the max width and height
# 	resized_images = [img.resize((max_width, max_height), Image.Resampling.LANCZOS) for img in images]

# 	# Calculate the total width and height for the combined image
# 	total_width = max_width * columns + separator_width * (columns - 1)
# 	total_height = max_height * rows + separator_width * (rows - 1)
# 	combined_image = Image.new('RGB', (total_width, total_height), color='white')

# 	# Place images in the specified grid
# 	x_offset = 0
# 	y_offset = 0
# 	for i, img in enumerate(resized_images):
# 		combined_image.paste(img, (x_offset, y_offset))
# 		if (i + 1) % columns == 0:  # Move to the next row after the last column
# 			x_offset = 0
# 			y_offset += img.height + separator_width
# 		else:  # Move to the next column
# 			x_offset += img.width + separator_width

# 	# Add numbers to each image for identification
# 	draw = ImageDraw.Draw(combined_image)
# 	try:
# 		font_size = (max_width + max_height) // 2 // 12
# 		font = ImageFont.load_default(size=font_size)
# 	except IOError:
# 		font = ImageFont.truetype("arial", 20)

# 	x_offset = 0
# 	y_offset = 0
# 	for i, img in enumerate(resized_images):
# 		text = str(i + 1)
# 		text_x = x_offset + 10
# 		text_y = y_offset + 10
# 		text_width, text_height = font_size, font_size
# 		font_color = get_contrasting_color(combined_image, text_x, text_y, text_width, text_height)
# 		draw.text((text_x, text_y), text, fill=font_color, font=font)
# 		if (i + 1) % columns == 0:
# 			x_offset = 0
# 			y_offset += img.height + separator_width
# 		else:
# 			x_offset += img.width + separator_width

# 	return combined_image

def concatenate_image(images, rows, columns, separator_width=10):
	# Calculate the max width and height of all provided images
	max_width = max(img.width for img in images)
	max_height = max(img.height for img in images)

	# Resize all images to a uniform size
	resized_images = [img.resize((max_width, max_height), Image.Resampling.LANCZOS) for img in images]

	# Calculate number of images needed
	total_images_needed = rows * columns
	padding_needed = total_images_needed - len(resized_images)

	# Pad with blank white images if needed
	if padding_needed > 0:
		blank_img = Image.new('RGB', (max_width, max_height), color='white')
		resized_images.extend([blank_img] * padding_needed)

	# Create the blank canvas for the grid
	total_width = max_width * columns + separator_width * (columns - 1)
	total_height = max_height * rows + separator_width * (rows - 1)
	combined_image = Image.new('RGB', (total_width, total_height), color='white')

	# Paste images into grid
	x_offset = 0
	y_offset = 0
	for i, img in enumerate(resized_images):
		combined_image.paste(img, (x_offset, y_offset))
		if (i + 1) % columns == 0:
			x_offset = 0
			y_offset += max_height + separator_width
		else:
			x_offset += max_width + separator_width

	# Draw labels
	# draw = ImageDraw.Draw(combined_image)
	# try:
	# 	font_size = (max_width + max_height) // 2 // 12
	# 	font = ImageFont.load_default(size=font_size)
	# except:
	# 	font = ImageFont.truetype("arial", 20)

	# x_offset = 0
	# y_offset = 0
	# for i in range(len(resized_images)):
	# 	text = str(i + 1)
	# 	text_x = x_offset + 10
	# 	text_y = y_offset + 10
	# 	text_width, text_height = font_size, font_size
	# 	font_color = get_contrasting_color(combined_image, text_x, text_y, text_width, text_height)
	# 	draw.text((text_x, text_y), text, fill=font_color, font=font)
	# 	if (i + 1) % columns == 0:
	# 		x_offset = 0
	# 		y_offset += max_height + separator_width
	# 	else:
	# 		x_offset += max_width + separator_width
    	# Draw labels
	draw = ImageDraw.Draw(combined_image)
	try:
		font_size = (max_width + max_height) // 2 // 12
		font = ImageFont.load_default(size=font_size)
	except:
		font = ImageFont.truetype("arial", 20)

	x_offset = 0
	y_offset = 0
	original_count = len(images)  # Only label original images
	for i in range(original_count):
		text = str(i + 1)
		text_x = x_offset + 10
		text_y = y_offset + 10
		text_width, text_height = font_size, font_size
		font_color = get_contrasting_color(combined_image, text_x, text_y, text_width, text_height)
		draw.text((text_x, text_y), text, fill=font_color, font=font)
		if (i + 1) % columns == 0:
			x_offset = 0
			y_offset += max_height + separator_width
		else:
			x_offset += max_width + separator_width

	return combined_image

from PIL import Image
# Load or create 6 images
images = [Image.open(f'01_pointing.png')]*7

# Concatenate into a 2x3 grid
result = concatenate_image(images, rows=2, columns=4)

# Save or display the result
result.save("output.jpg")