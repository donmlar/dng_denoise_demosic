
import shutil
import os


def backup(mark):
    if not os.path.isdir('../resultset/'+mark):
        os.makedirs('../resultset/'+mark)
        os.makedirs('../resultset/' + mark + '/network/')
        os.makedirs('../resultset/' + mark + '/loss/')
        os.makedirs('../resultset/' + mark + '/run/')
    if mark =="mix1":

        cover_files('../network', '../resultset/'+mark+'/network/')
        cover_files('../loss', '../resultset/'+mark+'/loss/')
        cover_files('../run', '../resultset/'+mark+'/run/')


def cover_files(source_dir, target_ir):
    for file in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file)

        if os.path.isfile(source_file):
            shutil.copy(source_file, target_ir)


if __name__ == '__main__':
    backup('mix1')