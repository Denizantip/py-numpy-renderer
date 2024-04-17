This repo was created only for educational purpose I wanted to understand how the rendering works. It do not have any practiacal sense. It slow as hell.
The naming of variables are not the best. But should be quiet clear.
Try to go inside the code and read it.
The most heavy computation part is shadows. First I've tried shadow map. but the algorithm is looks capricious for me. Thats why switched to Shadow Volume. Result much better and the algorithm is more general but much computationally intensive.

Small Documentation:
to render the Wavefron obj model need to perform few steps.
1. load model
   ```
   from core import Model
   some_model = Model.load_model(path_to_model)
   ```
2. (Optional step) Apply chain of transformations for model. (Or just one)
   ```
   from transformation import scale, translation, rotate
   some_model = some_model @ scale(factor) @ translation((x, y, z)) @ rotate((degrees_x, degrees_y, degrees_z))
   ```
3. Create light. (in Scene class will be used default light if not provided)
   ```
   from core import Light
   from lightning import Lightning
   
   light = Light(position,               #  position is (x, y, z) coordinates
                 light_type: Lightning,  # type of light (DIRECTIONAL_LIGHTNING, POINT_LIGHTNING, SPOT_LIGHTNING). Default is Lightning.POINT_LIGHTNING
                 center,                 # Coordinates to which it will directed (makes sense for spot and directional light)
   ) 
   ```
4. Camera MOTOR!!!
   ```python
   from core import Camera
   camera = Camera(position,                         # Nothing new
                   center,                           # Coordinated of point to which camera will "look"
                   fovy,                             # field of view in degrees (default is 90)
                   near,                             # near clip plane
                   far,                              # far clip plane
                   backface_culling: bool,           # is faces that "looks" not to camera will be skipped
                   projection_type: PROJECTION_TYPE  #  PERSPECTIVE or ORTHOGRAPHIC
   )
   ```
5. create Scene:
   ```python
   from core import Scene
   scene = Scene(camera, light, resolution:tuple)  # try to guess what resolution means. I think you can do it!
   ```
6. Add models to scene:
   ```python
   scene.add_model(some_model)
   scene.add_model(some_another_model)
   ...
   ```
7. Render it.
   ```python
   frame = scene.render()  #  the frame is numpy array (dtype=uint8) with shape (*resolution, 3)
   ```
8. Save/Convert/visualize the frame.
Some examples:
![Test](img/1.png)
![Test](img/2.png)
![Test](img/3.png)
![Test](img/4.png)
![Test](img/5.png)
![Test](img/6.png)
![Test](img/7.png)
![Test](img/8.png)
![Test](img/9.png)
![Test](img/0.png)
![Test](img/11.png)
![Test](img/12.png)
![Test](img/13.png)
![Test](img/14.png)
![Test](img/15.png)
![Test](img/16.png)
![Test](img/17.png)
![Test](img/18.png)
