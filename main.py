import os
from model import Model
from PIL import Image

authors = ['VanGogh']
path = lambda author: 'data/img/' + author + '/'
styles = {author: [path(author) + file for file in os.listdir(path(author))] for author in authors}
model = Model()
best, best_loss = model.run_style_transfer('data/img/Content/photo.jpg', styles['VanGogh'], 200)
result = Image.fromarray(best)
result.save('out.png')
