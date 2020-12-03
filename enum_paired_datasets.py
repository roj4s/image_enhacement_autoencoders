import os

def enum_paired_dirs(*dirs, scale=None):
    '''
        Rename all files contained in directory paths
        in dirs, to numbers (e.g 1,2,3.jpg) so files can
        be loaded more eficiently (lazy loading). First path
        in dirs is used as reference. If scale is not None
        it is expected that for all directories but the first
        in dirs, files contained have x1 as part of its name
        and scale as part of name for files in first directory
        in dirs.
    '''
    assert len(dirs) > 1, "At least two directory paths should be provided"
    ref = dirs[0]
    for i, file_name in enumerate(os.listdir(ref)):
        file_name, file_extension = os.path.splitext(file_name)
        ref_path = os.path.join(ref, f"{file_name}{file_extension}")
        new_ref_path = os.path.join(ref, f"{i}{file_extension}")
        print(f"{ref_path} => {new_ref_path}")
        os.rename(ref_path, new_ref_path)
        if scale is not None:
            file_name = file_name.replace('x1', scale)
            file_name = f"{file_name}{scale}"
        for _dir in dirs[1:]:
            _path = os.path.join(_dir, f"{file_name}{file_extension}")
            _new_path = os.path.join(_dir, f"{i}{file_extension}")
            os.rename(_path, _new_path)
            print(f"{_path} => {_new_path}")


if __name__ == "__main__":
    import sys
    dir_1 = sys.argv[1]
    dir_2 = sys.argv[2]
    scale = sys.argv[3]
    enum_paired_dirs(dir_1, dir_2, scale=scale)
