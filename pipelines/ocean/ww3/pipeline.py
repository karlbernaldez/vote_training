from pipelines.ocean.ww3.ingest.ingest_ww3 import main
from pipelines.ocean.ww3.transform.merge_forecast import merge_wave_transforms

def run_pipeline():
    main()
    merge_wave_transforms()
    print('WW3 pipeline active')
