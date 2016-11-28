
import os
import os.path
import convert_to_records
import numpy as np
import tensorflow as tf

def labelData():
    mugdata_ = os.listdir("./CV_Images/Mug")
    mugdata = [os.path.join("./CV_Images/Mug", img) for img in mugdata_]
    mugdatalabels = ["Mug" for i in mugdata]

    notmugdata_ = os.listdir("./CV_Images/Not_Mug")
    notmugdata = [os.path.join("./CV_Images/Not_Mug", img) for img in notmugdata_]
    notmugdatalabels = ["Not Mug" for i in notmugdata]

    mugdata.extend(notmugdata)
    mugdatalabels.extend(notmugdatalabels)

    filename_queue = tf.train.string_input_producer(mugdata) #  list of files to read

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.

    init_op = tf.initialize_all_variables()
    images = []
    with tf.Session() as sess:
        sess.run(init_op)
        
        # Start populating the filename queue.
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for i in range(1): #length of your filename list
            image = my_img.eval() #here is your image Tensor :)
            images.insert(0, image)

        coord.request_stop()
        coord.join(threads)

    images.reverse()
    return np.asarray(images), np.asarray(mugdatalabels)

def doConvert(data, labels, name="test"):
    
    convert_to_records.convert_to(data, labels, name)

