import gym
import common.atari_wrapper
import agents.dqn

import numpy
import time

import common.image


import models.atari_dqn.pacman.src.model
import models.atari_dqn.pacman.src.config

load_path = "./models/atari_dqn/pacman/"


input_shape     = (4,96,96)
outputs_count   = 9

model  = models.atari_dqn.pacman.src.model.Model(input_shape, outputs_count)

model.load(load_path)


layers          = [0, 3, 6, 9]
kernels_count   = [32, 32, 64, 64]



for l in range(len(layers)):
    images = []

    for kernel in range(kernels_count[l]):

        layer  = layers[l]

        print("processing layer ", layer, " kernel ", kernel)

        file_name = load_path + "kernel_visualisation/" + str(layer) + "_" + str(kernel) + ".png"

        result = model.kernel_visualise(layer, kernel, iterations=1000)

        result_color = numpy.zeros((3,input_shape[1], input_shape[2]))
        result_color[0] = result[0]
        result_color[1] = result[1]
        result_color[2] = result[2]

        images.append(result_color)

    file_name = load_path + "kernel_visualisation/" + str(layer) + ".png"

    common.image.multi_image_plot(images, file_name)