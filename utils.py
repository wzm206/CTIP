def waypoint_normalize(waypoint_ori, x_min, x_max, y_min, y_max):
    waypoint = waypoint_ori.clone()
    assert waypoint.shape[-1] == 2
    waypoint[...,0] = (waypoint[...,0]-x_min)/(x_max-x_min)
    waypoint[...,0] = waypoint[...,0]*2 - 1
    waypoint[...,1] = (waypoint[...,1]-y_min)/(y_max-y_min)
    waypoint[...,1] = waypoint[...,1]*2 - 1
    return waypoint

def waypoint_unnormalize(waypoint_ori, x_min, x_max, y_min, y_max):
    waypoint = waypoint_ori.clone()
    assert waypoint.shape[-1] == 2
    waypoint = (waypoint+1)/2
    
    waypoint[...,0] = waypoint[...,0]*(x_max-x_min)+x_min
    waypoint[...,1] = waypoint[...,1]*(y_max-y_min)+y_min
    return waypoint