import tensorflow as tf
from time import gmtime, strftime
from param import *
import csv

class TFLogger(object):
    def __init__(self, sess, var_list):
        self.sess = sess

        self.summary_vars = []
        self.var_list = var_list
        for var in var_list:
            tf_var = tf.Variable(0.)
            tf.summary.scalar(var, tf_var)
            self.summary_vars.append(tf_var)

        self.summary_ops = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(
            args.result_folder + args.model_folder + \
            strftime("%Y-%m-%d %H-%M-%S", gmtime()))
    
    def write_dict_log(self, val):
        file = args.result_folder + args.model_folder + strftime("%Y-%m-%d", gmtime())
        row = dict(zip(self.var_list,val))
        with open(file, "a", newline="") as f:
            for i in range(len(self.var_list)):
                f.write("%s,%s\n"%(self.var_list[i],val[i]))

            f.write("\n")

    def log(self, ep, values):
        assert len(self.summary_vars) == len(values)

        print(values)
        self.write_dict_log(values)

        #feed_dict = {self.summary_vars[i]: values[i] \
        #    for i in range(len(values))}
        
        #print(feed_dict)

        #summary_str = self.sess.run(
        #    self.summary_ops, feed_dict=feed_dict)
        
        #print(summary_str)

        #self.writer.add_summary(summary_str, ep)
        #self.writer.flush()
