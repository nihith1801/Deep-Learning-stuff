import os
import cv2
import random
import numpy as np
import glob as glob

class ImageProcessor:
    def __init__(self,image_size=(64,64)):
        #Initialize the image size
        self.image_size=image_size
    def load_image(self,file_path):
        image= cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image=cv2.resize(image, self.image_size)
        return image
    def apply_feature_masks(self,image):
        if image.dtype!=np.uint8:
            image=(image*255).astype(np.uint8) #Ensure 8 bit integer (0-255) for opencv

        #Enchancing contrast with CLAHE (Contrast Limiting adaptive equalization) to equalize brighntess through histogram
        clahe_convertor= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_image=clahe_convertor.apply(image)
        #Hnadling edge detection for morphed
        edges=cv2.Canny(enhanced_image, 100, 200) #Uses canny edge detection that happens internally through gaussian filter... ( )
        blended_image=cv2.addWeighted(enhanced_image, 0.8,edges,0.2,0)
        return blended_image.astype(np.float32)/255.0
    
    def augment_the_image(self,image):
        #Applying the random conversions
        if random.random()>0.5:
            image=cv2.flip(image,1) #Horizontal flip that's why 1 is given

        if random.random()>0.5:
            image=cv2.flip(image,0) #Vertical flip
        
        if random.random()>0.5:
            crop=int(self.image_size[0]*0.1)  #10 percent extra zoom karenge
            if crop>0:
                image=image[crop:-crop, crop:-crop] #Cropping x and y
                image=cv2.resize(image,self.image_size)
        return image
    
    def compute_class_weights(self,y_raw):
        #Calculating class biasness based on the no of images in each class
        unique_classes=np.unique(y_raw)
        num_classes=len(unique_classes)
        class_counts=np.bincount(y_raw)
        total_samples=len(y_raw)
        print(f"Total samples: {total_samples}")
        print(f"Class counts: {class_counts}")
        
        #Computing class weights through inverse frequency distribution..
        weights=total_samples/(num_classes*class_counts)
        print(f"Calculated class weights: {weights}")
        return weights

    
    def load_directory(self,image_directory):
        images=[]
        labels=[]
        image_classes= sorted([d for d in os.listdir(image_directory)
                               if os.path.isdir(os.path.join(image_directory,d))])
        
        image_classes_to_indexes={image_class: index for index,image_class in enumerate(image_classes)}
        print(f"Got these classes:{image_classes_to_indexes}")
        
        #Now looping and loading images
        for class_name in image_classes:
            folder_path= os.path.join(image_directory, class_name, '*') #Taking every type of paths
            image_count=0
            for image_path in glob.glob(folder_path):
                image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image=cv2.resize(image,self.image_size)
                images.append(image)
                labels.append(image_classes_to_indexes[class_name])
                image_count+=1

            print(f"Loaded {image_count} images from {class_name}")
        return np.array(images),np.array(labels),image_classes_to_indexes




if __name__=="__main__":
    image_processor=ImageProcessor(image_size=(64,64))
    train_dir="Deep-Learning-stuff/CNN/archive/MRI/Training"
    X_train, y_train, class_map=image_processor.load_directory(train_dir)
    print(f"\nTotal train images: {X_train.shape[0]}")
    print(f"\nImage Shape: {X_train.shape[1:]}")
    class_weights=image_processor.compute_class_weights(y_train)
