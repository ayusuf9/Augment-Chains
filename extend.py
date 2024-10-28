default_securities = []
if any(market_type.lower() == "developed market" for market_type in market_types_list):
    default_securities.extend(['Nvidia (2379504)', 'Qualcomm (2714923)'])
if any(market_type.lower() == "emerging market" for market_type in market_types_list):
    default_securities.extend(['Lenovo Group (6218089)', 'Infosys (6205122)'])