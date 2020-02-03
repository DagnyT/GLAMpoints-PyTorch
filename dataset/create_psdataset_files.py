import glob

ps_scenes_train = ['4', '11', '13' ,'14', '15', '16', '20', '23', '24', '30', '31', '34', '36', '41', '44', '49', '50', '51', '53', '65', '66', '67', '71', '74', '76','88', '89']
ps_scenes_test = [ '90', '91', '95']
root_dir = 'PS-Dataset/'

images_path = (sorted(glob.glob(root_dir + '/**/images_RGB/*.png', recursive=True)))
images_path = [x.replace(root_dir,'') for x in images_path]

scenes_used_train = { i : 0 for i in ps_scenes_train }

train_images, test_images = [], []

for i in images_path:
    scene = i.split('/')[0]
    if scene in ps_scenes_train:
        if scenes_used_train[scene]<58:
            train_images.append(i)
            scenes_used_train[scene]+=1

for i in images_path:
    scene = i.split('/')[0]
    if scene in ps_scenes_test:
        test_images.append(i)

with open('ps-dataset_train.txt', 'w') as f:
    for item in train_images: f.write("%s\n" % item)

with open('ps-dataset_test.txt', 'w') as f:
    for item in test_images: f.write("%s\n" % item)
