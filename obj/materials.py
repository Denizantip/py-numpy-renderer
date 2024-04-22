import numpy as np


class Material:
    """
    https://paulbourke.net/dataformats/mtl/

    Ka - defines the Ambient color of the material to be (r,g,b).
    Kd - defines the Diffuse color of the material to be (r,g,b).
    Ks - defines the Specular color of the material to be (r,g,b). This color shows up in highlights.
    Ke - defines the Emission color of the material to be (r,g,b).
    Pm - Metalness
    Pr - Roughness
    Ke - EMission

    d - defines the non-transparency of the material to be alpha. The default is 1.0 (not transparent at all).
    The quantities d and Tr are the opposites of each other, and specifying transparency or nontransparency is
    simply a matter of user convenience.

    Tr - defines the transparency of the material to be alpha. The default is 0.0 (not transparent at all).
    The quantities d and Tr are the opposites of each other, and specifying transparency or nontransparency
    is simply a matter of user convenience.
    Tf - transmission filter

    Ns - defines the shininess of the material to be s. Specular exponent.
    Ni - Optical density

    sharpness
    illum - denotes the illumination model used by the material.
    illum = 1 indicates a flat material with no specular highlights, so the value of Ks is not used.
    illum = 2 denotes the presence of specular highlights, and so a specification for Ks is required.

    map_Ka filename. During rendering values are multiplied by Ka.
    map_Kd filename. During rendering values are multiplied by Kd.
    map_Ks filename. During rendering values are multiplied by Ks.
    map_Ns filename. During rendering values are multiplied by Ns.
    map_d  filename. During rendering values are multiplied by d.
    map_bump filename. Normal map
    bump filename. Normal map
    norm filename. Normal map
    map_Pm filename. Metalness map
    map_Pr filename. Roughness map
    map_Ke filename. Emissoin color map
    disp filename. Height (displacement) map.
    names a file containing a texture map, which should just be an ASCII dump of RGB values;
    """
    Pm = 0.5
    Pr = 0.5
    Ka = np.array((0.3, 0, 0))  # ambient color
    Kd = np.array((0.8, 0.8, 0.8))  # diffuse color
    Ks = np.array((1., 1., 1.))  # specular color
    d = 1.0  # alpha
    Tr = 0  # alpha
    Ns = 64  # Ks exponent. Shininess factor [1 -1000]
    illum = 1  # n

    def __setattr__(self, key, value):
        if len(value) == 1:
            try:
                super().__setattr__(key, float(value[0]))
            except ValueError:
                super().__setattr__(key, value[0])
        else:
            super().__setattr__(key, np.array(value, dtype=np.float32))

    def __getattr__(self, item):
        map = {
            "diffuse": ("map_Kd", "Kd"),
            "ambient": ("map_Ka", "Ka"),
            "specular": ("map_Ks", "Ks"),
            "shininess": ("map_Ns", "Ns")
         }
        attr = map.get(item, None)
        if attr is not None:
            return super(self).__getattr__(attr[0])
        else:
            raise AttributeError("No such attribute", item)
