import torch
from torch.utils.data import DataLoader, Dataset
from proteins import Sample
import os

class dataset(Dataset):



    def __init__(self, res, path):
        self.res = res
        self.path = path
        self.maps = os.listdir(self.PATH)
        for name in self.maps:
            if name[-3:] != 'map':
                self.maps.remove(name)

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, i):
        map_path = os.path.join(self.path, self.maps[i])
        s1 = Sample(self.res, map_path)
        s1.decompose()
        tiles = s1.tiles
        tiles = [torch.from_numpy(t) for t in tiles]
        tiles = [t.unsqueeze(0).unsqueeze(0) for t in tiles]
        tiles = torch.cat(tiles, 0)
        return tiles

def collate_fn(tiles):
    tiles = torch.cat(tiles,0)
    return tiles


data = dataset(1.0, '../../../../data/1.0')
data_gen = DataLoader(data, batch_size=1, shuffle=True, 
        collate_fn=collate_fn)
