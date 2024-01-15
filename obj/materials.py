import numpy as np


class Material:
    """
    Ka - defines the Ambient color of the material to be (r,g,b).
    Kd - defines the Diffuse color of the material to be (r,g,b).
    Ks - defines the Specular color of the material to be (r,g,b). This color shows up in highlights.
    Pm - Metalness
    Pr - Roughness
    Ke - EMission

    d - defines the non-transparency of the material to be alpha. The default is 1.0 (not transparent at all).
    The quantities d and Tr are the opposites of each other, and specifying transparency or nontransparency is
    simply a matter of user convenience.

    Tr - defines the transparency of the material to be alpha. The default is 0.0 (not transparent at all).
    The quantities d and Tr are the opposites of each other, and specifying transparency or nontransparency
    is simply a matter of user convenience.

    Ns - defines the shininess of the material to be s.

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
    map_Pm filename. Metalness
    map_Pr filename. Roughness
    disp filename. Height map
    names a file containing a texture map, which should just be an ASCII dump of RGB values;
    """
    Pm = 1
    Pr = 0.1
    Ka = np.array((0.3, 0.3, 0.3))  # ambient
    Kd = np.array((0.8, 0.8, 0.8))  # diffuse
    Ks = np.array((1.0, 1.0, 1.0))  # specular
    d = 1.0  # alpha
    Tr = 0  # alpha
    Ns = 1.0  # s
    illum = 1  # n

    def __setattr__(self, key, value):
        if len(value) == 1:
            try:
                super().__setattr__(key, float(value[0]))
            except ValueError:
                super().__setattr__(key, value[0])
        else:
            super().__setattr__(key, np.array(value, dtype=np.float32))