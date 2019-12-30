from matplotlib import pyplot as plt
from matplotlib import gridspec

def make_rectange(num):
    width = int(num**0.5)

    while num%width != 0:
        width+= 1

    height = num//width

    return width, height


def multi_image_plot(images, file_name):
    image_height    = images[0].shape[1]
    image_width     = images[0].shape[2]

    width, height = make_rectange(len(images))


    fig = plt.figure(figsize=(width, height)) 

    gs = gridspec.GridSpec(height, width, wspace=0.05, hspace=0.05, top=1.0, bottom=0.0, left=0.0, right=1.0) 

    for i in range(len(images)):
        result_color = images[i].transpose((1, 2, 0))

        y = i//width
        x = i%width

        ax= plt.subplot(gs[y,x])
        ax.axis('off')
        ax.imshow(result_color)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
       

    plt.savefig(file_name, dpi = 150)
    