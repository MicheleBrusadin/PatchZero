from GTSRB_CNN.data_pre_proc import data_pre_proc_main
from GTSRB_CNN.data_pre_proc import load_data
from GTSRB_CNN.Train import train_main
from Patch_Attack_GTSRB.Attack import attack_main
from Defense.Defense import defense_main

if __name__ == '__main__':
    img_size = (100, 100)
    min_size = (40, 40)
    defense_batch_size = 8
    defense_epochs = 100

    save_dir = 'data'
    data_dir = 'datasets/original/gtsrb'
    plots_dir = 'plots'
    train_csv_filename = 'Train.csv'
    test_csv_filename = 'Test.csv'
    train_data_file_name = 'train.npy'
    test_data_file_name = 'test.npy'
    weights_file_name = 'result.weights.h5'
    model_filename = 'modelimagenet.keras'

    train_path = f'{save_dir}/{train_data_file_name}'
    test_path = f'{save_dir}/{test_data_file_name}'
    weights_path = f'{save_dir}/{weights_file_name}'

    output_train_dir = f'{save_dir}/train'
    output_test_dir = f'{save_dir}/test'
    test_images_dir = f'{output_test_dir}/images'
    test_masks_dir = f'{output_test_dir}/masks'
    train_images_dir = f'{output_train_dir}/images'
    train_masks_dir = f'{output_train_dir}/masks'

    # data_pre_proc_main(data_dir=data_dir, train_csv_filename=train_csv_filename, test_csv_filename=test_csv_filename,
    #                    save_dir=save_dir, img_size=img_size, min_size=min_size,
    #                    train_filename=train_data_file_name, test_filename=test_data_file_name)

    # train_main(train_path=train_path, test_path=test_path, image_size=img_size, weights_path=weights_path,
    #            save_dir=save_dir)

    attack_main(image_size=img_size, train_path=train_path, test_path=test_path, weights_path=weights_path,
                save_dir=save_dir)

    defense_main(train_dir=output_train_dir, test_dir=output_test_dir, output_dir=save_dir, plots_dir=plots_dir,
                 model_filename=model_filename, batch_size=defense_batch_size, epochs=defense_epochs)


