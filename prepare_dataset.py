import os
from shutil import copyfile

src_dir = 'data/speech_commands_dataset'
dst_dir = 'data/speech_commands_prepared'

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)
for f in ['train', 'validation', 'test']:
    if not os.path.exists(os.path.join(dst_dir, f)):
        os.mkdir(os.path.join(dst_dir, f))

validation_file = os.path.join(src_dir, 'validation_list.txt')
testing_file = os.path.join(src_dir, 'testing_list.txt')

data_files = [validation_file, testing_file]
data_dirs = ['validation', 'test']
def prepare_ds():
    all_fnames = []
    for i, phase in enumerate(('validation', 'test')):
        with open(data_files[i], newline='\n') as f:
            fnames = f.read().splitlines()
            print(len(fnames))
        all_fnames.extend(fnames)
        for fname in fnames:
            if fname.endswith('.wav'):
                cls_dir = os.path.join(dst_dir, data_dirs[i], os.path.split(fname)[0])
                if not os.path.exists(cls_dir):
                    os.mkdir(cls_dir)
                copyfile(src=os.path.join(src_dir, fname), dst=os.path.join(dst_dir, data_dirs[i], fname))
            else:
                print("Skipping file: {}".format(fname))

    else:
        dirs = [d.name for d in os.scandir(src_dir) if d.is_dir()]
        fnames = [os.path.join(d, fname)
                  for d in dirs
                  for fname in os.listdir(os.path.join(src_dir, d))
                  if fname.endswith('.wav')
                  and os.path.join(d, fname) not in all_fnames]
        for fname in fnames:
            if fname.endswith('.wav'):
                cls_dir = os.path.join(dst_dir, 'train', os.path.split(fname)[0])
                if not os.path.exists(cls_dir):
                    os.mkdir(cls_dir)
                copyfile(src=os.path.join(src_dir, fname), dst=os.path.join(dst_dir, 'train', fname))
            else:
                print("Skipping file: {}".format(fname))

if __name__ == '__main__':
    prepare_ds()
