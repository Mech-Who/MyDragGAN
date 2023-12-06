# pip install lpips
# python lpips_2dirs.py -dir0 ./input_images -dir1 ./output_images


def learned_perceptual_image_patch_similarity():
    """
    Learned Perceptual Image Patch Similarity （LPIPS）：
    计算编辑图像与目标图像之间的感知相似度。
    LPIPS使用预训练的深度学习模型来度量图像之间的感知差异，
    较低的LPIPS值表示编辑图像更接近目标图像。
    """
    pass


if __name__ == "__main__":
    print("lpips test:")

    import argparse
    import os
    import lpips

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_image_dir', type=str, default='./input_images')
    parser.add_argument('-o', '--output_image_dir', type=str, default='./output_images')
    parser.add_argument('-v', '--version', type=str, default='0.1')

    opt = parser.parse_args()

    # Initializing the model
    loss_fn = lpips.LPIPS(net='alex', version=opt.version)

    # the total list of images
    files = os.listdir(opt.input_image_dir)
    i = 0
    total_lpips_distance = 0
    average_lpips_distance = 0
    for file in files:
        try:
            # load images\
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.input_image_dir, file)))
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.output_image_dir, file)))

            img0_exist = os.path.exists(os.path.join(opt.input_image_dir, file))
            img1_exist = os.path.exists(os.path.join(opt.output_image_dir, file))

            if (img0_exist, img1_exist):
                i += 1

            # compute distance
            current_lpips_distance = loss_fn.forward(img0, img1)
            total_lpips_distance += current_lpips_distance

            print(f"{file:s}: {current_lpips_distance:.3f}")

        except Exception as e:
            print(e)

    average_lpips_distance = float(total_lpips_distance) / i

    print(f"The processed images is {i} and the average_lpips_distance is : {average_lpips_distance:.3f}")
