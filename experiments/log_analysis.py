import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_log(log_file):
    # read txt
    f = open(log_file)
    line = f.readline()

    while line:
        line = f.readline()
        if 'input image' in line.lower():
            print(line)
            processed_line=line.split('center_cropped_')[-1].split('.tif')[0].split('X')[0]
            processed_line = processed_line.split('_')
            processed_line = [int(_) for _ in processed_line]
            print(processed_line)
            image_size = np.prod(processed_line)
            print(image_size)

        if 'ssim' in line.lower():
            ssim = line.split('|')[2]
            print('SSIM', ssim.strip())

        if 'psnr' in line.lower():
            psnr = line.split('|')[2]
            print('PSNR', psnr.strip())

    f.close()

    return image_size, float(ssim), float(psnr)

if __name__ == '__main__':
    results_dir = '/home/heyz/code/z-vision/results'
    results_dirs = os.listdir(results_dir)
    results_dirs.sort()
    idx = results_dirs.index('20210629_20_49_43')
    dirs_to_analysis = results_dirs[idx:]

    psnr_list = list()
    ssim_list = list()
    data_size_list = list()

    for dir in dirs_to_analysis:
        size, ssim, psnr = read_log(os.path.join(results_dir,dir,'log.txt'))
        data_size_list.append(size)
        ssim_list.append(ssim)
        psnr_list.append(psnr)

    # sort
    data_size_sort_index = np.argsort(data_size_list)
    data_size_list.sort()
    ssim_list_sort = [ssim_list[i] for i in data_size_sort_index]
    psnr_list_sort = [psnr_list[i] for i in data_size_sort_index]

    plt.plot(data_size_list, psnr_list_sort, 'ro')
    plt.show()

    with open("datasize_psnr_ssim.csv", 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow(data_size_list)
        writer.writerow(psnr_list_sort)
        writer.writerow(ssim_list_sort)
