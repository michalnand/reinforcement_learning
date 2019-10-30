


class Linear():
    def __init__(self, iterations, start_value = 1.0, end_value = 0.1):
        self.decay = (start_value - end_value)*1.0/iterations   
        self.start_value = start_value
        self.end_value = end_value

        self.epsilon = self.start_value

    def process(self):
        if self.epsilon > self.end_value:
            self.epsilon-= self.decay
        return self.epsilon

    def get(self):
        return self.epsilon

    def get_start(self):
        return self.start_value

    def get_end(self):
        return self.end_value


class Exponential():
    def __init__(self, q = 0.999999, start_value = 1.0, end_value = 0.1):
        self.q = q
        self.start_value = start_value
        self.end_value = end_value

        self.epsilon = self.start_value

    def process(self):
        if self.epsilon > self.end_value:
            self.epsilon = self.epsilon*self.q
        return self.epsilon

    def get(self):
        return self.epsilon

    def get_start(self):
        return self.start_value

    def get_end(self):
        return self.end_value