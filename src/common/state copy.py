import numpy
from matplotlib import pyplot as plt


class State():
    def __init__(self, input_shape, target_shape = (80, 80), frame_stacking = 1):

        self.frame_stacking = frame_stacking

        self.input_width     = input_shape[0]
        self.input_height    = input_shape[1]
        self.input_depth     = input_shape[2]

        self.width_ratio  = self.input_width//target_shape[0]
        self.height_ratio = self.input_height//target_shape[1]

        self.width_crop    = self.input_width  - self.width_ratio*target_shape[0]
        self.height_crop   = self.input_height - self.height_ratio*target_shape[1]

        self.slice_shape    = (self.input_depth, target_shape[0], target_shape[1])
        self.shape          = (1, frame_stacking*self.input_depth, target_shape[0], target_shape[1])

        self.clear()

        print("scale ratio ", self.width_ratio, self.height_ratio)
        print("crop ", self.width_crop, self.height_crop)
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


    def update_state(self, raw_state):

       

        if self.width_crop%2 != 0:
            width_odd = -1
        else:
            width_odd = 0

        if self.height_crop%2 != 0:
            height_odd = -1
        else:
            height_odd = 0

        start_x = self.width_crop//2
        end_x   = self.input_width - self.width_crop//2 + width_odd
        start_y = self.height_crop//2
        end_y   = self.input_height - self.height_crop//2 + height_odd


        swaped_state  = numpy.rollaxis(raw_state, 2, 0)
        cropped_state = swaped_state[::, start_x:end_x, start_y:end_y]/255.0
        scaled_state  = cropped_state[::, ::self.height_ratio, ::self.width_ratio]


        if scaled_state.shape != self.slice_shape:
            print("ERROR : State::update_state : size mismatch, expecting ", self.slice_shape, ", get :", scaled_state.shape)
        else:
            for i in reversed(range(self.frame_stacking-1)):
                self.slices[i+1] = self.slices[i].copy()
            self.slices[0] = numpy.array(scaled_state, dtype = numpy.float32).copy()
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
