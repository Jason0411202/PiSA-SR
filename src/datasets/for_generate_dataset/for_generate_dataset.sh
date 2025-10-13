# HC18 training set (bicubic 4x)
python -m src.datasets.for_generate_dataset.for_generate_dataset \
--dataset_dir ../HC18/training_set \
--output_dir src/datasets/for_generate_dataset/outputs/bicubic_4x/training_set \
--upscale 4 \
--gt_width 512 \
--gt_height 512 \

# HC18 test set (bicubic 4x)
python -m src.datasets.for_generate_dataset.for_generate_dataset \
--dataset_dir ../HC18/test_set \
--output_dir src/datasets/for_generate_dataset/outputs/bicubic_4x/test_set \
--upscale 4 \
--gt_width 512 \
--gt_height 512 \

# HC18 training set (gaussian noise [10, 60])
python -m src.datasets.for_generate_dataset.for_generate_dataset \
--dataset_dir ../HC18/training_set \
--output_dir src/datasets/for_generate_dataset/outputs/gaussian_noise/training_set \
--degradation_file for_generate_dataset/gaussian_noise.yml \
--upscale 1 \
--gt_width 512 \
--gt_height 512 \

# HC18 test set (gaussian noise [10, 60])
python -m src.datasets.for_generate_dataset.for_generate_dataset \
--dataset_dir ../HC18/test_set \
--output_dir src/datasets/for_generate_dataset/outputs/gaussian_noise/test_set \
--degradation_file for_generate_dataset/gaussian_noise.yml \
--upscale 1 \
--gt_width 512 \
--gt_height 512 \

# HC18 training set (gaussian noise [10, 30])
python -m src.datasets.for_generate_dataset.for_generate_dataset \
--dataset_dir ../HC18/training_set \
--output_dir src/datasets/for_generate_dataset/outputs/gaussian_noise_10_30/training_set \
--degradation_file for_generate_dataset/gaussian_noise_10_30.yml \
--upscale 1 \
--gt_width 512 \
--gt_height 512 \

# HC18 test set (gaussian noise [10, 30])
python -m src.datasets.for_generate_dataset.for_generate_dataset \
--dataset_dir ../HC18/test_set \
--output_dir src/datasets/for_generate_dataset/outputs/gaussian_noise_10_30/test_set \
--degradation_file for_generate_dataset/gaussian_noise_10_30.yml \
--upscale 1 \
--gt_width 512 \
--gt_height 512 \

# HC18 training set (speckle noise)
python -m src.datasets.for_generate_dataset.for_generate_dataset \
--dataset_dir ../HC18/training_set \
--output_dir src/datasets/for_generate_dataset/outputs/speckle_noise/training_set \
--degradation_file for_generate_dataset/speckle_noise.yml \
--upscale 1 \
--gt_width 512 \
--gt_height 512 \

# HC18 test set (speckle noise)
python -m src.datasets.for_generate_dataset.for_generate_dataset \
--dataset_dir ../HC18/test_set \
--output_dir src/datasets/for_generate_dataset/outputs/speckle_noise/test_set \
--degradation_file for_generate_dataset/speckle_noise.yml \
--upscale 1 \
--gt_width 512 \
--gt_height 512 \

# HC18 training set (gaussian blur)
python -m src.datasets.for_generate_dataset.for_generate_dataset \
--dataset_dir ../HC18/training_set \
--output_dir src/datasets/for_generate_dataset/outputs/gaussian_blur/training_set \
--degradation_file for_generate_dataset/gaussian_blur.yml \
--upscale 1 \
--gt_width 512 \
--gt_height 512 \

# HC18 test set (gaussian blur)
python -m src.datasets.for_generate_dataset.for_generate_dataset \
--dataset_dir ../HC18/test_set \
--output_dir src/datasets/for_generate_dataset/outputs/gaussian_blur/test_set \
--degradation_file for_generate_dataset/gaussian_blur.yml \
--upscale 1 \
--gt_width 512 \
--gt_height 512 \

# HC18 training set (complex)
python -m src.datasets.for_generate_dataset.for_generate_dataset \
--dataset_dir ../HC18/training_set \
--output_dir src/datasets/for_generate_dataset/outputs/complex/training_set \
--degradation_file for_generate_dataset/complex.yml \
--upscale 4 \
--gt_width 512 \
--gt_height 512 \

# HC18 test set (complex)
python -m src.datasets.for_generate_dataset.for_generate_dataset \
--dataset_dir ../HC18/test_set \
--output_dir src/datasets/for_generate_dataset/outputs/complex/test_set \
--degradation_file for_generate_dataset/complex.yml \
--upscale 4 \
--gt_width 512 \
--gt_height 512 \

