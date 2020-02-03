import os
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:

    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def add_scalars_to_tensorboard(self, type, epoch, iter, total_loss, values):
        rmse, mee, mae, mma_1, mma_3, mma_5, acceptable_homography, repeatability = values

        self.writer.add_scalar(type+'/Loss', total_loss, iter)
        self.writer.add_scalar(type+'/Repeatability', repeatability, iter)
        self.writer.add_scalar(type+'/MMA@1', mma_1, iter)
        self.writer.add_scalar(type+'/MMA@3', mma_3, iter)
        self.writer.add_scalar(type+'/MMA@5', mma_5, iter)
        self.writer.add_scalar(type+'/acceptable_homography', acceptable_homography, iter)

    def add_images_to_tensorboard(self, img, name):

        self.writer.add_image('images/'+name, img, 0)

class FileLogger:
    "Log text in file."
    def __init__(self, path):
        self.path = os.path.join(path, 'log.txt')
        self.init_logs()

    def init_logs(self):

        text_file = open(self.path, "w")
        text_file.close()

    def log_string(self, string):
        """Stores log string in log file."""
        text_file = open(self.path, "a")
        text_file.write(str(string)+'\n')
        text_file.close()