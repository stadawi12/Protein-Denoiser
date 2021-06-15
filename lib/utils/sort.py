import os, random, math
import numpy as np

class Maps:

    rnd = None

    def __init__(self, path_global):
        assert path_global[-1] == '/'
        self.path_global = path_global
        self.ls = os.listdir(path_global)
        self.get_maps()
        self.get_rest()
        self.get_contents()


    def get_maps(self):
        # Obtains maps found in global directory
        self.maps = []
        for name in self.ls:
            if '.map' in name:
                self.maps.append(name)
        self.maps.sort()

    def set_random(self):
        """
            generates a randomly shuffled list from 0 to
            the number of maps found in global directory
            and saves it to a static class variable
        """
        Maps.rnd = np.arange(0, len(self.maps))
        random.shuffle(Maps.rnd)

    def get_rest(self):
        """
            obtains the rest of files (except for maps) in the
            global directory, ignores the .gitkeep file.
        """
        lst = []
        for name in self.ls:
            if name not in self.maps:
                lst.append(name)
        lst.remove('.gitkeep')
        lst = [float(l) for l in lst]
        lst.sort()
        lst = [str(l) for l in lst]
        self.rest = lst
        
    def del_rest(self):
        """
            deletes everything that isn't a .map file
            in the global directory (ignores .gitkeep file)
        """
        if self.rest != []:
            paths = [self.path_global + r for r in self.rest]
            contents = [os.listdir(p) for p in paths]
            condition = False
            for content in contents:
                if content == []:
                    condition = True
            assert condition, "WARNING! Folder(s) are not empty"
            for path in paths:
                os.removedirs(path)
            self.ls = os.listdir(self.path_global)
            self.get_maps()
            self.get_rest()
            self.get_contents()

    def mkdirs(self, number_of_dirs):
        """
            makes a number of directories in the global
            directory, e.g. if you want 3 directories
            to be created you can do,
              self.mkdirs(3) 
            and you would get three dirs named:
              '1.0/', '2.0/', '3.0/'.
            need to ensure the global directory doesn't 
            contain any directories first.
        """
        assert self.rest == [], "self.rest is not empty"
        dirs = []
        for i in range(number_of_dirs):
            dirs.append('{}.0'.format(i+1))
        for d in dirs:
            os.mkdir(self.path_global + d)
        self.ls = os.listdir(self.path_global)
        self.get_maps()
        self.get_rest()
        self.get_contents()

    def get_contents(self):
        # obtains the contents of the sub directories of
        # the global directory, stores the in 
        # self.rest_contents
        dir_paths = [self.path_global + b + '/' for b in self.rest]
        dir_contents = [os.listdir(dir_path) 
                for dir_path in dir_paths]
        self.rest_contents = dir_contents

    def divide(self, size):
        """
            generates a list of subdivisions of maps into
            sub-directories, for example if we have 8 maps
            and want them to be divided into 3 sub-
            directories then we would store them in a 
            format of [3, 3, 2] which is what this function
            would return.
        """
        self.flatten()
        # a/b = c
        a = len(self.maps)
        b = size
        if a % b == 0:
            b_true = b
        else:
            b_true = b - 1
        lst = []
        for i in range(b_true):
            lst.append(math.ceil(a/b))
        if b_true != size:
            lst.append(a - b_true*math.ceil(a/b))
            
        return lst

    def sort(self):
        """
            Sorts our maps randomly into empty sub-directories.
            if we have 8 maps and 2 sub-directories in the 
            the global directory, this function would shuffle
            the maps and store them into the two sub-directories
            evenly ([4, 4]).
        """
        assert self.rnd.all() != None,\
        "self.rnd has not been defined"
        assert len(self.rnd) == len(self.maps), \
            "Length of self.rnd is not same as length of self.maps"
        batch_size = math.ceil(len(self.maps) / len(self.rest))
        iter_list = self.divide(len(self.rest))
        self.maps.sort()
        count = 0
        for j, batch_size in enumerate(iter_list):
            for i in range(count, count+batch_size):
                index = self.rnd[i]
                path_original = self.path_global + self.maps[index]
                path_dest = self.path_global + self.rest[j] +\
                        '/' + self.maps[index]
                os.rename(path_original, path_dest)
            count += batch_size
        self.ls = os.listdir(self.path_global)
        self.get_maps()
        self.get_rest()
        self.get_contents()

    def flatten(self):
        """
            This function takes all the maps out of the 
            sub-directories and places them in the global
            directory.
        """
        dir_paths = [self.path_global + b + '/' for b in self.rest]
        dir_contents = [os.listdir(dir_path) 
                for dir_path in dir_paths]
        map_paths = []
        for i in range(len(dir_paths)):
            paths = []
            for j, name in enumerate(dir_contents[i]):
                paths.append(dir_paths[i] + name)
            map_paths.append(paths)

        map_dest = []
        for i in range(len(map_paths)):
            temp_list = []
            for name in dir_contents[i]:
                temp_list.append(self.path_global + name)
            map_dest.append(temp_list)

        for i in range(len(map_paths)):
            for j in range(len(map_paths[i])):
                os.rename(map_paths[i][j], map_dest[i][j])
        self.ls = os.listdir(self.path_global)
        self.get_maps()
        self.get_rest()
        self.get_contents()

    def move_to_bad(self, map_id):
        path_map = self.path_global + map_id
        path_bad = self.path_global[:-1] + 'badMaps/' + map_id
        os.rename(path_map, path_bad)
        self.ls = os.listdir(self.path_global)
        self.get_maps()
        self.get_rest()
        self.get_contents()

    def move_to_good(self, map_id):
        path_map = self.path_global + map_id
        path_bad = self.path_global[:-1] + 'badMaps/' + map_id
        os.rename(path_bad, path_map)
        self.ls = os.listdir(self.path_global)
        self.get_maps()
        self.get_rest()
        self.get_contents()

    def move_all_from_bad(self):

        # create path to badMaps (e.g. 1.0+badMaps=1.0badMaps)
        path_bad = self.path_global[:-1] + 'badMaps/'

        # Get contents of badMaps 
        contents = os.listdir(path_bad)
        
        # Remove .gitkeep from contents to prevent deletion
        contents.remove('.gitkeep')

        # Wrap over move_to_good method
        for map_id in contents:
            self.move_to_good(map_id)
            
        # Update attributes the directory
        self.ls = os.listdir(self.path_global)
        self.get_maps()
        self.get_rest()
        self.get_contents()
        




if __name__ == '__main__':
    m1 = Maps('../../data/1.0/')
    m2 = Maps('../../data/2.0/')
    m1.del_rest()
    m2.del_rest()
    m2.set_random()
    m1.mkdirs(6)
    m2.mkdirs(6)
    m1.sort()
    m2.sort()
    print(m1.maps == m2.maps)
    print(len(m1.maps))
    print(len(m2.maps))
