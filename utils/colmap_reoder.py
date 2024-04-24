import sqlite3
import os
import argparse

def read_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM images")
    images_tuples = c.fetchall()

    c.execute("SELECT * FROM cameras")
    cameras_tuples = c.fetchall()

    return cameras_tuples, images_tuples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str)
    parser.add_argument("--database_filename", type=str, default="database.db")
    parser.add_argument("")
    args = parser.parse_args()

    dir_db = os.path.join(args.datadir, 'db.db')
    file_images = os.path.join(args.datadir, 'sparse_learned', 'images.txt')

    # Read db file to dictionary.
    cam_dict, img_dict = read_db(dir_db)

    # Parse images.txt file line-by-line.
    image_list = []
    with open(file_images, 'r') as f:
        lines = f.readlines()
        #import pdb; pdb.set_trace()
        for line in lines:
            if line != '\n':
                image_list.append(line)
                #print(line.split(' ')[-1].replace('\n', ''))
    os.system("mv {} {}".format(file_images, file_images.replace('images.txt', 'images_sorted.txt')))
    
    with open(file_images, 'w') as f:
    #with open(os.path.join(args.datadir, 'sparse_learned', 'debug.txt'), 'w') as f:
        for data in img_dict:
            print(data[1])
            for img_data in image_list:
                img_name = img_data.split(' ')[-1].replace('\n', '')

                if img_name == data[1]:
                    idx_split = img_data.find(' ', 1)
                    img_data_new = str(data[0]) + ' ' + img_data[idx_split+1:]
                    f.write(img_data_new)
                    f.write('\n')
                    
