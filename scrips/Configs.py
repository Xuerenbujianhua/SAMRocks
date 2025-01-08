class Config(object):
    Classes_rocks = [
        'background',
        'Hornblende',#
        'Pyroxene',#
        'Olivine',#
        'Anhydrite',#
        'Quartz',#
        'Dolomite',#
        'Plagioclase',#
        'Muscovite',#
        'Calcite', #
        'Aphanocrystalline',#
        'Cordierite',#
        'um',#
    ]
    # class color
    Palette_rocks = [
        [0, 0, 0],  # background
        [34, 139, 34],  # Hornblende (Green)#
        [139, 0, 0],  # Pyroxene (Dark Red)#
        [0, 100, 0],  # Olivine (Olive Green)#
        [255, 228, 196],  # Anhydrite (Beige)#
        [169, 169, 169],  # Quartz (Light Gray)#
        [210, 180, 140],  # Dolomite (Tan)#
        [135, 206, 250],  # Plagioclase (Light Blue)#
        [255, 222, 173],  # Muscovite (Navajo White)#
        [255, 248, 220],  # Calcite (Cornsilk)#
        [255, 182, 193],  # Aphanocrystalline (Light Pink)#
        [75, 0, 130],  # Cordierite (Indigo)#
        [0, 0, 100],  # um_50 (Dark Blue) #
    ]

    Model_Info = {
        'densenet121': '../checkpoints/densenet121.pth',
        'mobilenet_v2': '../checkpoints/mobilenet_v2.pth',
        'resnet18': '../checkpoints/resnet18.pth',
        'resnet152': '../checkpoints/resnet152.pth',
        'resnet50': '../checkpoints/resnet50.pth',
    }