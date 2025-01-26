# Adversarial Patch Attack


"""
Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""
if __name__ == "__main__":


    import argparse
    import csv
    
    import numpy as np
    
    from patch_utils import*
    from utils import*
    import pandas as pd
   
    
    
    import os
    from GTSRB_CNN.train import get_model, r2_keras
    
    import tensorflow as tf
    from tensorflow.keras.losses import SparseCategoricalCrossentropy
    from PIL import Image


    


    
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
    parser.add_argument('--train_size', type=int, default=30, help="number of training images")
    parser.add_argument('--test_size', type=int, default=30, help="number of test images")
    parser.add_argument('--noise_percentage', type=float, default=0.1, help="percentage of the patch size compared with the image size")
    parser.add_argument('--probability_threshold', type=float, default=0.9, help="minimum target probability")
    parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
    parser.add_argument('--max_iteration', type=int, default=1000, help="max iteration")
    parser.add_argument('--target', type=int, default=30, help="target label")
    parser.add_argument('--epochs', type=int, default=20, help="total epoch")
    parser.add_argument('--data_dir', type=str, default='data', help="dir of the dataset")

    

    parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
    parser.add_argument('--GPU', type=str, default='0', help="index pf used GPU")
    parser.add_argument('--log_dir', type=str, default='patch_attack_log.csv', help='dir of the log')
    args = parser.parse_args()

    # visualize the patch effect on the model




    def visualize_patch_effect(image, patched_image, output_path_original, output_path_patched):
        """
        Visualize and save the effect of an adversarial patch on an image.
        """
        # GTSRB dataset normalization values
        mean = np.array([0.3337, 0.3064, 0.3171])  # Mean values for R, G, B channels
        std  = np.array([0.2672, 0.2564, 0.2629])  # Standard deviation values for R, G, B channels
    	# save the patched image as a tensor
        tf.io.write_file('image_tensor.tfrecord', tf.io.serialize_tensor(patched_image))
        

        #save he patched image as a numpy array
        np.save('patched_image.npy', patched_image.numpy().squeeze(0))
        

        # this conversion to png is not working
        image = preprocess_and_denormalize(image, mean, std)
        patched_image = preprocess_and_denormalize(patched_image, mean, std)

        
        


        # Save the original and patched images using OpenCV
        cv2.imwrite(output_path_original, image)
        cv2.imwrite(output_path_patched, patched_image)

        print(f"Saved original image to {output_path_original}")
        print(f"Saved patched image to {output_path_patched}")

    def preprocess_and_denormalize(image, mean, std):
        """
        Preprocesses and denormalizes an image for visualization.
        Args:
            image: TensorFlow tensor of shape (height, width, 3) or (3, height, width).
            mean: NumPy array of mean values for R, G, B.
            std: NumPy array of standard deviation values for R, G, B.
        Returns:
            Denormalized image in uint8 format ready for saving.
        """
        # Convert to numpy and squeeze extra dimensions
        
        image = tf.squeeze(image).numpy()

        # Ensure proper value range before scaling
        # image = np.clip(image, 0, 1)

        # Convert to 0-255 and uint8 format
        image = (image * 255.0).round().astype(np.uint8)
    
        return image


    def patch_attack(image, applied_patch, mask, target, probability_threshold, model, lr=1, max_iteration=100):
        """
        Perform a patch attack via optimization.

        Args:
            image: Input image tensor (normalized).
            applied_patch: Initial adversarial patch (NumPy array).
            mask: Binary mask indicating where the patch is applied.
            target: Target class index for the adversarial attack.
            probability_threshold: Threshold for the target class probability to stop optimization.
            model: TensorFlow model.
            lr: Learning rate for patch optimization.
            max_iteration: Maximum number of optimization iterations.

        Returns:
            perturbated_image: Final adversarial image (NumPy array).
            applied_patch: Final adversarial patch (NumPy array).
        """
        # Convert applied_patch and mask to tensors
        applied_patch = tf.convert_to_tensor(applied_patch, dtype=tf.float32)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

        
        mask = tf.expand_dims(mask, axis=0)


        
        
        applied_patch = tf.expand_dims(applied_patch, axis=0)
        target_probability = 0
        count = 0

        # Perform patch attack optimization
        while target_probability < probability_threshold and count < max_iteration:
            count += 1

            with tf.GradientTape() as tape:
                tape.watch(applied_patch)

                # Apply the patch to the image
                

                perturbated_image = tf.multiply(mask, applied_patch) + tf.multiply(1 - mask, image)
                perturbated_image = tf.clip_by_value(perturbated_image, -3.0, 3.0)  # Clamp to valid range
                

                # Forward pass through the model
                classification_output, detection_output = model(perturbated_image, training=False)
                log_softmax_output = tf.nn.log_softmax(classification_output, axis=1)
                target_log_softmax = log_softmax_output[0, target]

            # Compute gradients of the loss with respect to the patch
            patch_grad = tape.gradient(target_log_softmax, applied_patch)

            # Update the patch using the gradient
            applied_patch += lr * patch_grad
            applied_patch = tf.clip_by_value(applied_patch, -3.0, 3.0)  # Clamp to valid range

            # Test the patch

            perturbated_image = tf.multiply(mask, applied_patch) + tf.multiply(1 - mask, image)
            perturbated_image = tf.clip_by_value(perturbated_image, -3.0, 3.0)

            # Compute target class probability
            classification_output, detection_output = model(perturbated_image, training=False)
            
            softmax_output = tf.nn.softmax(classification_output, axis=1)
            target_probability = softmax_output[0, target].numpy()

            
        # Convert tensors back to NumPy arrays
        perturbated_image = perturbated_image.numpy()
        applied_patch = applied_patch.numpy()

        return perturbated_image, applied_patch


    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    # Load the model
    model = get_model()
    loss = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss={"classification": loss, "regression": "mse"},
                  metrics={"classification": "acc", "regression": r2_keras},
                  loss_weights={"classification": 5, "regression": 1})
    model.load_weights("weights/weights.h5")
    print("Model is loaded")
    # Load the datasets
    
    train_df = pd.read_csv("data/Train.csv")
    test_df = pd.read_csv("data/Test.csv")
    # Preprocess the data
    images, bboxes, labels = preprocess_dataset(train_df, "data", img_size=(30, 30))
    test_images, test_bboxes, test_labels = preprocess_dataset(test_df, "data", img_size=(30, 30))
    print("Data is preprocessed")
    # Create TensorFlow dataloader
    train_loader = create_dataloader(images, bboxes, labels, batch_size=1)
    test_loader = create_dataloader(test_images, test_bboxes, test_labels, batch_size=1)
    print("DataLoader is created")
    # Test the accuracy of model on trainset and testset
    trainset_acc, test_acc = test(model, train_loader), test(model, test_loader)
    print('Accuracy of the model on clean trainset and testset is {:.3f}% and {:.3f}%'.format(100*trainset_acc, 100*test_acc))

    # Initialize the patch
    patch = patch_initialization(args.patch_type, image_size=(30, 30, 3), noise_percentage=args.noise_percentage)
    # print('The shape of the patch is', patch.shape)

    with open(args.log_dir, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_success", "test_success"])

    best_patch_epoch, best_patch_success_rate = 0, 0

    # Generate the patch
    for epoch in range(args.epochs):
        train_total, train_actual_total, train_success = 0, 0, 0

        for idx, (image, label,*_) in enumerate(train_loader):
            print(idx)
            if idx == 150:
                break
            train_total += label.shape[0]
            
            assert image.shape[0] == 1, "Only one picture should be loaded at a time."
            
            # Forward pass 
            classification_output, detection_output = model(image, training=False)
            original_prediction = tf.argmax(classification_output, axis=1)

            if original_prediction.numpy()[0] != args.target:  # Skip correctly predicted target
                train_actual_total += 1

                # Generate patch and mask
                applied_patch, mask, x_location, y_location = mask_generation(
                    args.patch_type, patch, image_size=(30, 30, 3)
                )

                # Run patch attack optimization
                perturbated_image, applied_patch = patch_attack(
                    image.numpy(), applied_patch, mask, args.target,
                    args.probability_threshold, model, args.lr, args.max_iteration
                )

                # Convert the perturbed image back to a tensor

                
                perturbated_image = tf.convert_to_tensor(perturbated_image, dtype=tf.float32)

                # Forward pass with the perturbed image
                classification_output, detection_output  = model(perturbated_image, training=False)
                patched_prediction = tf.argmax(classification_output , axis=1)
                print("Patched Prediction:", patched_prediction)
                


                
                   

                if patched_prediction.numpy()[0] == args.target:
                    train_success += 1
                    #save original and successful patched images 

                    
                    if epoch ==0:
                        visualize_patch_effect(
                        image=image,  # Use the original image
                        patched_image=perturbated_image,  # Use the patched image
                    
                        output_path_original=f"training_pictures_GTSRB/original/epoch_{epoch}_sample_{idx}.png",
                        output_path_patched=f"training_pictures_GTSRB/patched/epoch_{epoch}_sample_{idx}.png",
                        )
                
                patch_shape = tf.shape(patch)
                # print("Patch shape:", patch_shape)
                x_location_end = x_location + patch_shape[0]
                y_location_end = y_location + patch_shape[1]	

                patch = applied_patch[0, x_location:x_location_end, y_location:y_location_end,:]
                # print("Patch shape:", patch.shape)  # Should match applied_patch slice shape
                 
             
               
       
        mean = [0.3337, 0.3064, 0.3171]  # Mean values for R, G, B channels
        std  = [0.2672, 0.2564, 0.2629]  # Standard deviation values for R, G, B channels


    #     # plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
    #     plt.savefig("training_pictures/" + str(epoch) + " patch.png")
    #     print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success / train_actual_total))
    #     train_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
    #     print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success_rate))
    #     test_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
    #     print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))

    #     # Record the statistics
    #     with open(args.log_dir, 'a') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([epoch, train_success_rate, test_success_rate])

    #     if test_success_rate > best_patch_success_rate:
    #         best_patch_success_rate = test_success_rate
    #         best_patch_epoch = epoch
    #         plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
    #         plt.savefig("training_pictures/best_patch.png")

    #     # Load the statistics and generate the line
    #     log_generation(args.log_dir)

    # print("The best patch is found at epoch {} with success rate {}% on testset".format(best_patch_epoch, 100 * best_patch_success_rate))