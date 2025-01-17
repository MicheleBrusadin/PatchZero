# Adversarial Patch Attack
# Created by Junbo Zhao 2020/3/17

"""
Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    import torchvision
    from torchvision import models

    import argparse
    import csv
    import os
    import numpy as np

    from patch_utils import*
    from utils import*
    from torchvision.transforms.functional import to_pil_image

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
    parser.add_argument('--train_size', type=int, default=800, help="number of training images")
    parser.add_argument('--test_size', type=int, default=200, help="number of test images")
    parser.add_argument('--noise_percentage', type=float, default=0.03, help="percentage of the patch size compared with the image size")
    parser.add_argument('--probability_threshold', type=float, default=0.9, help="minimum target probability")
    parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
    parser.add_argument('--max_iteration', type=int, default=1000, help="max iteration")
    parser.add_argument('--target', type=int, default=859, help="target label")
    parser.add_argument('--epochs', type=int, default=20, help="total epoch")
    parser.add_argument('--data_dir', type=str, default='datasets\imgNet\imagenet1k_valid_dataset', help="dir of the dataset")

    

    parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
    parser.add_argument('--GPU', type=str, default='0', help="index pf used GPU")
    parser.add_argument('--log_dir', type=str, default='patch_attack_log.csv', help='dir of the log')
    args = parser.parse_args()

    # visualize the patch efffect on the model
    def visualize_patch_effect(image, patched_image, original_prediction, patched_prediction, class_names, output_path_original, output_path_patched):
        """
        Visualize and save the effect of an adversarial patch on an image.
        """
        # Denormalize the images
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = denormalize(image.squeeze(), mean, std).cpu()  # Ensure image is on the CPU
        patched_image = denormalize(patched_image.squeeze(), mean, std).cpu()

        # Convert to PIL images for matplotlib compatibility
        from torchvision.transforms.functional import to_pil_image
        original_image = to_pil_image(image)
        patched_image = to_pil_image(patched_image)

        # Save the original and patched images separately
        original_image.save(output_path_original)
        patched_image.save(output_path_patched)

    def denormalize(image, mean, std):
        mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
        std = torch.tensor(std).view(3, 1, 1).to(image.device)
        denormalized = image * std + mean
        return denormalized.clamp(0, 1)  # Ensure values are within [0, 1]

    # Patch attack via optimization
    # According to reference [1], one image is attacked each time
    # Assert: applied patch should be a numpy
    # Return the final perturbated picture and the applied patch. Their types are both numpy
    def patch_attack(image, applied_patch, mask, target, probability_threshold, model, lr=1, max_iteration=100):
        model.eval()
        applied_patch = torch.from_numpy(applied_patch)
        mask = torch.from_numpy(mask)
        target_probability, count = 0, 0
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        while target_probability < probability_threshold and count < max_iteration:
            count += 1
            # Optimize the patch
            perturbated_image = Variable(perturbated_image.data, requires_grad=True)
            per_image = perturbated_image
            per_image = per_image.cuda()
            output = model(per_image)
            target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
            target_log_softmax.backward() # Calculate the gradient, backward pass
            patch_grad = perturbated_image.grad.clone().cpu() # Get the gradient of the patch, so to adjust and increase target probability
            perturbated_image.grad.data.zero_()
            applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor) # Update the patch
            applied_patch = torch.clamp(applied_patch, min=-3, max=3)
            # Test the patch
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
            perturbated_image = perturbated_image.cuda()
            output = model(perturbated_image)
            target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
        perturbated_image = perturbated_image.cpu().numpy()
        applied_patch = applied_patch.cpu().numpy()
        
        return perturbated_image, applied_patch

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    # Load the model
    model = models.resnet50(pretrained=True).cuda()
    model.eval()

    # Load the datasets
    train_loader, test_loader = dataloader(args.train_size, args.test_size, args.data_dir, args.batch_size, args.num_workers, 1000)

    # Test the accuracy of model on trainset and testset
    trainset_acc, test_acc = test(model, train_loader), test(model, test_loader)
    print('Accuracy of the model on clean trainset and testset is {:.3f}% and {:.3f}%'.format(100*trainset_acc, 100*test_acc))

    # Initialize the patch
    patch = patch_initialization(args.patch_type, image_size=(3, 224, 224), noise_percentage=args.noise_percentage)
    print('The shape of the patch is', patch.shape)

    with open(args.log_dir, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_success", "test_success"])

    best_patch_epoch, best_patch_success_rate = 0, 0

    # Generate the patch
    for epoch in range(args.epochs):
        train_total, train_actual_total, train_success = 0, 0, 0
        for idx, (image, label) in enumerate(train_loader):
            train_total += label.shape[0]
            assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            _, original_prediction  = torch.max(output.data, 1)
            if  original_prediction[0].data.cpu().numpy() != args.target: # removed to use also correct predictions original_prediction[0] != label                
                train_actual_total += 1
                applied_patch, mask, x_location, y_location = mask_generation(args.patch_type, patch, image_size=(3, 224, 224))
                perturbated_image, applied_patch = patch_attack(image, applied_patch, mask, args.target, args.probability_threshold, model, args.lr, args.max_iteration)
                perturbated_image = torch.from_numpy(perturbated_image).cuda()
                output = model(perturbated_image)
                _, patched_prediction = torch.max(output.data, 1)
                if patched_prediction[0].data.cpu().numpy() == args.target:
                    train_success += 1
                    #save original and successful patched images 
                    visualize_patch_effect(
                    image=image,  # Use the original image
                    patched_image=perturbated_image,  # Use the patched image
                    original_prediction=original_prediction.item(),
                    patched_prediction=patched_prediction.item(),
                    class_names=train_loader.dataset.classes,
                    output_path_original=f"training_pictures/original/epoch_{epoch}_sample_{idx}.png",
                    output_path_patched=f"training_pictures/patched/epoch_{epoch}_sample_{idx}.png",
                    )
                patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
                 
             
               
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        

        plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
        plt.savefig("training_pictures/" + str(epoch) + " patch.png")
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success / train_actual_total))
        train_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch, 100 * train_success_rate))
        test_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
        print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, 100 * test_success_rate))

        # Record the statistics
        with open(args.log_dir, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_success_rate, test_success_rate])

        if test_success_rate > best_patch_success_rate:
            best_patch_success_rate = test_success_rate
            best_patch_epoch = epoch
            plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
            plt.savefig("training_pictures/best_patch.png")

        # Load the statistics and generate the line
        log_generation(args.log_dir)

    print("The best patch is found at epoch {} with success rate {}% on testset".format(best_patch_epoch, 100 * best_patch_success_rate))