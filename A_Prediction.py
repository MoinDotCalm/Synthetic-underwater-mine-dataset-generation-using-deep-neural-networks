from sklearn.metrics import *
import tensorflow.compat.v1 as tf
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

start = time.time()

try:
    
    # Path of  training images
    train_path = r'C:\Users\moini\OneDrive\Desktop\major project\part A\train'
    if not os.path.exists(train_path):
        print("No such directory")
        raise Exception
    # Path of testing images
    dir_path = r'C:\Users\moini\OneDrive\Desktop\major project\part A\test'
    if not os.path.exists(dir_path):
        print("No such directory")
        raise Exception
    
    # Walk though all testing images one by one
    for root, dirs, files in os.walk(dir_path):
        for name in files:

            print("")
            image_path = name
            filename = dir_path +'\\' +image_path
            print(filename)
            image_size=128
            num_channels=3
            images = []
        
            if os.path.exists(filename):
                
                # Reading the image using OpenCV
                image = cv2.imread(filename)
                # Resizing the image to our desired size and preprocessing will be done exactly as done during training
                image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                images.append(image)
                images = np.array(images, dtype=np.uint8)
                images = images.astype('float32')
                images = np.multiply(images, 1.0/255.0) 
            
                # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
                x_batch = images.reshape(1, image_size,image_size,num_channels)

                # Let us restore the saved model 
                sess = tf.Session()
                # Step-1: Recreate the network graph. At this step only graph is created.
                saver = tf.train.import_meta_graph('models/trained_model.meta')
                # Step-2: Now let's load the weights saved using the restore method.
                saver.restore(sess, tf.train.latest_checkpoint('./models/'))

                # Accessing the default graph which we have restored
                graph = tf.get_default_graph()

                # Now, let's get hold of the op that we can be processed to get the output.
                # In the original network y_pred is the tensor that is the prediction of the network
                y_pred = graph.get_tensor_by_name("y_pred:0")

                ## Let's feed the images to the input placeholders
                x= graph.get_tensor_by_name("x:0") 
                y_true = graph.get_tensor_by_name("y_true:0") 
                y_test_images = np.zeros((1, len(os.listdir(train_path)))) 


                # Creating the feed_dict that is required to be fed to calculate y_pred 
                feed_dict_testing = {x: x_batch, y_true: y_test_images}
                result=sess.run(y_pred, feed_dict=feed_dict_testing)
                # Result is of this format [[probabiliy_of_classA probability_of_classB ....]]
                print(result)

                # Convert np.array to list
                a = result[0].tolist()
                r=0

                # Finding the maximum of all outputs
                max1 = max(a)
                index1 = a.index(max1)
                predicted_class = None

                # Walk through directory to find the label of the predicted output
                count = 0
                for root, dirs, files in os.walk(train_path):
                    for name in dirs:
                        if count==index1:
                            predicted_class = name
                        count+=1

                # If the maximum confidence output is largest of all by a big margin then
                # print the class or else print a warning
                for i in a:
                    if i!=max1:
                        if max1-i<i:
                            r=1                           
                if r ==0:
                    print(predicted_class)
                else:
                    print("Could not classify with definite confidence")
                    print("Maybe:",predicted_class)

            # If file does not exist
            else:
                print("File does not exist")
                
except Exception as e:
    print("Exception:",e)

# Calculate execution time
end = time.time()
dur = end-start
print("")
if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")

y_true = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#[AR0,HF1,HG7,IG3,MF9,MGB12,MU2,MU5,MU6,MU8,MU10,MUGF4,SGB11,TG13]

#f1_score
x=a
for i in range(len(a)):
    x[i]=round(a[i])
accuracy = f1_score(y_true, x)
print("\nf1 score for the model: ",accuracy)


#MSE
accuracy =  mean_squared_error(y_true, a)
print("\nMSE for the model: ",accuracy)
#####

from sklearn.metrics import confusion_matrix

#gt = [1,1,2,2,1,0]
#pd = [1,1,1,1,2,0]

cm = confusion_matrix(y_true, x)

#rows = gt, col = pred

#compute tp, tp_and_fn and tp_and_fp w.r.t all classes
tp_and_fn = cm.sum(1)
tp_and_fp = cm.sum(0)
tp = cm.diagonal()

precision = tp / tp_and_fp
recall = tp / tp_and_fn
from operator import truediv
import numpy as np

tp = np.diag(cm)
prec = list(map(truediv, tp, np.sum(cm, axis=0)))
rec = list(map(truediv, tp, np.sum(cm, axis=1)))
print ('Precision: {}\nRecall: {}'.format(prec, rec))

#confusion_matrix

accuracy =  confusion_matrix(y_true, x)
print("\nconfusion matrix for the model:\n ",accuracy)


# Assuming you have the confusion matrix stored as "confusion_mat"
confusion_mat = confusion_matrix(y_true, x)

# Convert the confusion matrix to a NumPy array
confusion_mat_np = np.array(confusion_mat)


labels = np.unique(np.concatenate((y_true,x)))

# Create a figure and axis
fig, ax = plt.subplots()
im = ax.imshow(confusion_mat_np, cmap='Blues')

# Add labels, title, and color bar
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_title('Confusion Matrix')
plt.colorbar(im)

# Loop over data dimensions and add text annotations
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, confusion_mat[i, j], ha='center', va='center', color='black')

plt.show()

