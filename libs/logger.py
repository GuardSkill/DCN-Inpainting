import tensorflow as tf
import numpy as np
import scipy.misc
from io import BytesIO, StringIO  # Python 3.x


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        # self.writer = tf.summary.FileWriter(log_dir)   # older tf version
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # summary = tf.summary(value=[tf.summary.Value(tag=tag, simple_value=value)])
        with self.writer.as_default():
            tf.summary.scalar(tag, value,step=step)
            # self.writer.add_summary(summary, step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            # img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
            #                            height=img.shape[0],
            #                            width=img.shape[1])
            img_sum = tf.summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])

            # Create a Summary value
            # img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))
            img_summaries.append(tf.summary.Value(tag='%s/%d' % (tag, i), image=img_sum))
        # Create and write Summary
        # summary = tf.Summary(value=img_summaries)
        with self.writer.as_default():
            summary = tf.summary(value=img_summaries,step=step)
            # self.writer.add_summary(summary, step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        with self.writer.as_default():
            summary = tf.summary.histogram(tag, hist, collections=None, name=None)
            self.writer.add_summary(summary, step)
            self.writer.flush()