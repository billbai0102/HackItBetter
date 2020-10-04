import requests
import numpy as np

def get_image_by_id(id):
    request = "https://res.cloudinary.com/billbai0102/image/upload/v1595200550/hackitbetter/"+ id
    response = requests.get(request)
    image = response.content
    image = np.asarray(image)

    return image