import numpy
from matplotlib import pyplot as plt

import cv2

class State():
    def __init__(self, input_shape, target_shape = (96, 96), frame_stacking = 1, channel_swap = True):

        self.frame_stacking = frame_stacking
        self.target_shape   = target_shape
        self.channel_swap   = channel_swap

        self.input_width     = input_shape[0]
        self.input_height    = input_shape[1]
        self.input_depth     = input_shape[2]
       
        self.slice_shape    = (self.input_depth, self.target_shape[0], self.target_shape[1])
        self.shape          = (1, frame_stacking*self.input_depth, self.target_shape[0], self.target_shape[1])

        self.clear()

        print("slice_shape ", self.slices[0].shape)
        print("state_shape ", self.get_shape())


    def clear(self):
        self.state = numpy.zeros(self.shape)
        self.slices = []
        for i in range(0, self.frame_stacking):
            self.slices.append(numpy.zeros(self.slice_shape, dtype = numpy.float32))

    def get_shape(self):
        return self.state.shape

    def get(self):
        return self.state


    def update_state(self, frame):
        state = numpy.reshape(frame, frame.shape).astype(numpy.float32)/255.0
        resized = cv2.resize(state, self.target_shape)

        if self.channel_swap:
            swaped = numpy.moveaxis(resized, 2, 0)
        else:
            swaped = resized


        if swaped.shape != self.slice_shape:
            print("ERROR : State::update_state : size mismatch, expecting ", self.slice_shape, ", get :", swaped.shape, " original : ", frame.shape)
        else:
            for i in reversed(range(self.frame_stacking-1)):
                self.slices[i+1] = self.slices[i].copy()
            self.slices[0] = numpy.array(swaped).copy()
            self.state[0] = numpy.concatenate(self.slices, axis = 0)  


    def show(self):
        shape = self.state.shape
        image = numpy.zeros((shape[2], shape[3], 3))

        for ch in range(3):
            for y in range(shape[2]):
                for x in range(shape[3]):
                    image[y][x][ch] = self.state[0][ch][y][x]


        plt.imshow(image, interpolation='none')
        plt.show()
