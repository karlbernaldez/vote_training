from pipelines.ocean.ww3.transform.wave_period import transform_wave_period
from pipelines.ocean.ww3.transform.wave_direction import transform_wave_direction

def merge_wave_transforms():
    transform_wave_period()
    transform_wave_direction()
    print('WW3 transform orchestration active')
