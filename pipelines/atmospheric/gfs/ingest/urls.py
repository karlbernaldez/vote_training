def build_download_url(yyyy: str, mm: str, dd: str, hh: str, fhr: str) -> str:
    """Build NOAA NOMADS GFS filtered GRIB2 download URL."""
    return (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        f"?dir=%2Fgfs.{yyyy}{mm}{dd}%2F{hh}%2Fatmos"
        f"&file=gfs.t{hh}z.pgrb2.0p25.f{fhr}"
        "&var_UGRD=on"
        "&var_VGRD=on"
        "&lev_10_m_above_ground=on"
        "&subregion="
        "&toplat=50"
        "&leftlon=100"
        "&rightlon=180"
        "&bottomlat=-5"
    )
