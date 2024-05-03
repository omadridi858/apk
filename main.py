from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
import os
import numpy as np
from kivy.uix.progressbar import ProgressBar
# Designate Our .kv design file 
Builder.load_file('interface.kv')
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.factory import Factory
import matplotlib.pyplot as plt
import seaborn as sns
import activity
import cv2
from kivy.uix.image import Image

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg 
from kivy.garden.matplotlib.backend_kivy import FigureCanvas
import matplotlib
from kivy.graphics.texture import Texture

class PlotPopup(Popup):
    def __init__(self, fig, root, **kwargs):
        super(PlotPopup, self).__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (root.width / 1.2, root.height / 1.2)  # Set your desired size for the popup
        self.title = 'Plot'

        # Convert Matplotlib figure to a numpy array
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())

        # Convert RGBA to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # Flip the image vertically
        image = np.flipud(image)

        # Create a Texture object
        texture = Texture.create(size=(image.shape[1], image.shape[0]))
        texture.blit_buffer(image.tostring(), colorfmt='bgr', bufferfmt='ubyte')

        # Create an Image widget and set its texture
        self.image = Image(texture=texture)
        self.add_widget(self.image)
matplotlib.use('Agg')

class DirChooser(FileChooserListView):
    def __init__(self, **kwargs):
        super(DirChooser, self).__init__(**kwargs)
        self.dirselect = True







class MyLayout(Widget):

    def open_popup(self, fig):
        # Convert Matplotlib figure to a numpy array
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())

        # Convert RGBA to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # Display the image using OpenCV
        cv2.imshow('Plot100', image)
        cv2.waitKey(0)  # Wait for any key press to close the window
        cv2.destroyAllWindows()
    def plot_accuracy_loss(self, history):
        fig = plt.figure(figsize=(10, 5))

        plt.subplot(221)
        plt.plot(history.history['accuracy'], 'bo--', label="acc")
        plt.plot(history.history['val_accuracy'], 'ro--', label="val_acc")
        plt.title("train_acc vs val_acc")
        plt.ylabel("accuracy")
        plt.xlabel("epochs")
        plt.legend()

        plt.subplot(222)
        plt.plot(history.history['loss'], 'bo--', label="loss")
        plt.plot(history.history['val_loss'], 'ro--', label="val_loss")
        plt.title("train_loss vs val_loss")
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.legend()

        popup = PlotPopup(fig,self)
        popup.open()

    def testplot(self, history):
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)
        
        
        sns.heatmap(history[0], cbar=True, xticklabels=history[1], yticklabels=history[1], annot_kws={"size": 10},fmt='g', annot=True, cmap='Blues', ax=ax)

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        popup = PlotPopup(fig,self)
        popup.open()

    def selected(self, path, is_dir):
        print(path)

    def load_data_in_thread(self, path, is_dir):
        dataset_path=path
        # Automatically create a list of class names from the subdirectories
        subdirectories = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]

        class_names = subdirectories
        print(class_names)
        # Create a dictionary to map class names to labels
        class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

        # Number of classes
        nb_classes = len(class_names)


        IMAGE_SIZE = (32, 32)


        #Loading the Data

        def load_data(dataset_path):
            print("Loading {}".format(dataset_path))

            images = []
            labels = []

            # Iterate through each folder corresponding to a category
            for folder in os.listdir(dataset_path):
                label = class_names_label[folder]
                total_images = len(os.listdir(os.path.join(dataset_path, folder)))
                # Iterate through each image in our folder
                for i, file in enumerate(os.listdir(os.path.join(dataset_path, folder))):

                    # Get the path name of the image
                    img_path = os.path.join(os.path.join(dataset_path, folder), file)

                    # Load the image with error handling
                    try:
                        image = cv2.imread(img_path)
                        if image is None:
                            print(f"Error loading image: {img_path}")
                            continue
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, IMAGE_SIZE)
                    except Exception as e:
                        print(f"Error processing image: {img_path}, {e}")
                        continue
                    
                    # Append the image and its corresponding label to the output
                    images.append(image)
                    labels.append(label)
                    progress_value = (i + 1) / total_images
                    print(f'Loading {dataset_path} {folder}  ({i + 1} /{total_images})',progress_value)
                 

            images = np.array(images, dtype='float32')
            labels = np.array(labels, dtype='int32')

            return images, labels

        images, labels = load_data(dataset_path)
        global train_images, train_labels,test_images, test_labels,cl
        cl=class_names
        (train_images, train_labels), (test_images, test_labels)=(images, labels), (images, labels)
        print("complete")

    def traincnn(self):
        history1 = activity.start(1, train_images, train_labels, test_images, test_labels, cl, self)
        self.plot_accuracy_loss(history1)
        

    def trainmlp(self):
        history1 = activity.start(2, train_images, train_labels, test_images, test_labels, cl, self)
        self.plot_accuracy_loss(history1)
    def testcnn(self):
        history2 = activity.start(3, train_images, train_labels, test_images, test_labels, cl, self)
        
        self.testplot(history2)

    def testmlp(self):
        history2 = activity.start(4, train_images, train_labels, test_images, test_labels, cl, self)
        self.testplot(history2)
class AwesomeApp(App):
    def build(self):
        return MyLayout()

if __name__ == '__main__':
    AwesomeApp().run()
