from datetime import datetime


def build_gfs_wind_url(run_dt: datetime, run_hour: str, forecast_hour: str) -> str:
    yyyy = run_dt.strftime('%Y')
    mm = run_dt.strftime('%m')
    dd = run_dt.strftime('%d')
    return (
        'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl'
        f'?dir=%2Fgfs.{yyyy}{mm}{dd}%2F{run_hour}%2Fatmos'
        f'&file=gfs.t{run_hour}z.pgrb2.0p25.f{forecast_hour}'
        '&var_UGRD=on'
        '&var_VGRD=on'
        '&lev_10_m_above_ground=on'
        '&subregion='
        '&toplat=50'
        '&leftlon=100'
        '&rightlon=180'
        '&bottomlat=-5'
    )
