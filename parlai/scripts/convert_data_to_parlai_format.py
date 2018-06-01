    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: convert_data should be passed opt not Parser ]')
        opt = opt.parse_args()

