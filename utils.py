def eplased_time_since(start_time):
    import time
    curret_time = time.time()
    seconds = int(curret_time - start_time)
    hours = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    time_str = '{:0>2d}h{:0>2d}min{:0>2d}s'.format(hours, minutes, seconds)
    return time_str


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)