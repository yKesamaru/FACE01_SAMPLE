# If you want to get only face recognition

Don't forget to set `headless = True` on `config.ini`.

## Result
```bash

[2022-09-27 19:20:48,174] [face01lib.load_priset_image] [simple.py] [INFO] Loading npKnown.npz
菅義偉 
         similarity              99.1% 
         coordinate              (138, 240, 275, 104) 
         time                    2022,09,27,19,20,49,835926 
         output                   
 -------

麻生太郎 
         similarity              99.6% 
         coordinate              (125, 558, 261, 422) 
         time                    2022,09,27,19,20,49,835926 
         output                   
 -------
 ...
```

The face images are output to the `/output/` folder.
![](https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/img/PASTE_IMAGE_2022-07-20-07-36-26.png)

Try `example/benchmark_CUI.py` in the same examples folder.
You can see the profile result.
Your browser will automatically display a very cool and easy to use graph using `snakeviz`.

```bash
snakeviz restats
```

![](https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/img/PASTE_IMAGE_2022-07-20-07-23-21.png)

Whole example source code is [here](../example/benchmark_CUI.py).

```python
# Import FACE01 library
from face01lib.Core import Core
from face01lib.Initialize import Initialize
```

```python
"""Set the number of playback frames.
If you just want to try FACE01 a bit, you can limit the number of frames it loads."""

if __name__ == '__main__':
    main(exec_times = 5)
```

```python

def main(exec_times: int = 50):

    # Initialize
    CONFIG: Dict =  Initialize().initialize()

    # Make generator
    gen = Core().common_process(CONFIG)

    # Repeat 'exec_times' times
    for i in range(1, exec_times):

        # Call __next__() from the generator object
        frame_datas_array = gen.__next__()

        for frame_datas in frame_datas_array:
            
            for person_data in frame_datas['person_data_list']:
                if not person_data['name'] == 'Unknown':
                    print(
                        person_data['name'], "\n",
                        "\t", "similarity\t\t", person_data['percentage_and_symbol'], "\n",
                        "\t", "coordinate\t\t", person_data['location'], "\n",
                        "\t", "time\t\t\t", person_data['date'], "\n",
                        "\t", "output\t\t\t", person_data['pict'], "\n",
                        "-------\n"
                    )
```
