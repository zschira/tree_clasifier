def allowable_site(site: str):
    if site in ["OSBS", "MLBS"]:
        return site
    else:
        raise Exception("Unexpected site")

def allowable_data_type(data_type: str):
    if data_type in ["HSI", "CHM", "LAS", "RGB"]:
        return data_type
    else:
        raise Exception("Unexpected data type")
